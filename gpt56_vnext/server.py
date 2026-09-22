from __future__ import annotations

import asyncio
import logging
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, urlsplit
import uuid

from .benchmark import DEFAULT_TIER_COUNTS, MAX_PACKAGE_BYTES
from .catalog import BenchmarkCatalog
from . import __version__
from .agent_tasks import KIT, task_prompt
from .detector import DetectorSession
from .directory_lock import exclusive_directory
from .transport import AsyncTransport
from .errors import AppError, RequestError
from .executor import runtime_options, detection_options
from .presets import EndpointPresets, estimate_plan
from .schedule import SingleRunSchedule
from .store import SQLiteStateStore
from .updates import ProgramUpdates
from .utils import canonical_json, integer, strict_json_loads, normalize_site_group
from .security import SecretGuard

WEB_ROOT = Path(__file__).with_name("web")
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
          "/assets/ui.js": ("ui.js", "text/javascript; charset=utf-8"),
          "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/assets/workbench.js": ("workbench.js", "text/javascript; charset=utf-8"),
          "/assets/i18n.js": ("i18n.js", "text/javascript; charset=utf-8"),
          "/assets/style.css": ("style.css", "text/css; charset=utf-8")}


class AppState:
    def __init__(self, root, locale="zh-CN", *, bundled=True):
        if locale not in {"zh-CN", "en"}:
            raise AppError("unsupported_language")
        self.locale = locale
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with ExitStack() as resources:
            resources.enter_context(exclusive_directory(self.root))
            self.store = SQLiteStateStore(self.root / "state.sqlite3")
            resources.callback(self.store.close)
            self.store.interrupt_active_sessions()
            from . import BUNDLED_BASELINES
            self.catalog = BenchmarkCatalog(self.root / "benchmarks", Path(__file__).with_name("baselines") / BUNDLED_BASELINES if bundled else None)
            self.presets = EndpointPresets(self.store)
            self.updates = ProgramUpdates(self.root / "updates", self.store)
            self.active = {}
            self._closed = False
            self._closing = False
            self.rate_gates = {}
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
            self.thread.start()
            resources.callback(self._stop_loop)
            self.schedule = SingleRunSchedule(self.store, self.scheduled_run)
            self.log_maintenance_task = self.call(self._start_log_maintenance())
            self._resources = resources.pop_all()
            asyncio.run_coroutine_threadsafe(self.resume_updated_schedule(), self.loop)

    async def _start_log_maintenance(self):
        return asyncio.create_task(self._maintain_event_logs())

    async def _maintain_event_logs(self):
        while True:
            pending = asyncio.create_task(asyncio.to_thread(self.store.prune_events))
            try:
                removed = await asyncio.shield(pending)
            except asyncio.CancelledError:
                # A queued SQLite write must finish before shutdown closes the store.
                await asyncio.gather(pending, return_exceptions=True)
                raise
            except Exception:
                logging.getLogger(__name__).exception("Event retention failed")
                removed = 0
            # Catch up in bounded transactions without blocking the request loop.
            await asyncio.sleep(1 if removed == 1_000 else 300)

    async def resume_updated_schedule(self):
        resume_schedule = self.store.document('settings', 'resume_schedule_after_update')
        pending = self.store.document('settings', 'pending_baseline_updates')
        if not resume_schedule and not pending:
            return
        while self.updates.busy() and not self._closed:
            await asyncio.sleep(.5)
        if not self._closed:
            if pending:
                try:
                    await self.install_baselines(pending)
                except AppError:
                    pass  # Pending request stays visible and can be retried by the user.
            if resume_schedule:
                self.store.delete_document('settings', 'resume_schedule_after_update')
                await self.schedule.resume_after_update()

    async def prepare_exit(self):
        if self.active or (self.schedule.status() or {}).get('enabled') or self.updates.busy():
            raise AppError('finish_work_before_exit',status=409)
        self._closing = True

    async def install_baselines(self, requested):
        if not isinstance(requested, list) or len(requested) > 32:
            raise AppError('invalid_catalog_request')
        self.store.put_document('settings', 'pending_baseline_updates', requested)
        installed = [await self.catalog.install(p.get('id'), p.get('version')) for p in requested]
        defaults = self.store.document('settings', 'default_benchmarks') or {}
        for p in installed:
            defaults[p['mode']] = {'id':p['id'], 'version':p['version']}
        self.store.put_document('settings', 'default_benchmarks', defaults)
        self.store.delete_document('settings', 'pending_baseline_updates')
        return {'installed':len(installed)}

    def call(self, coroutine, timeout=30):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout)
        except FutureTimeout:
            future.cancel()
            raise

    def connection(self, body):
        key = body.get("key", "")
        if body.get("endpoint_id"):
            preset, saved = self.presets.connection(body["endpoint_id"])
            return {"base_url": preset["base_url"], "allow_insecure": preset["allow_insecure"],
                    "request_model": body.get("request_model") or preset["model"]}, key or saved
        return {key: body.get(key) for key in ("base_url", "allow_insecure", "request_model")}, key

    async def models(self, body):
        connection, key = self.connection(body)
        if not isinstance(key, str) or not key:
            raise AppError('credential_required')
        transport = AsyncTransport()
        try:
            return await transport.models(connection['base_url'], key,
                                          allow_insecure=connection.get('allow_insecure') is True)
        finally:
            await transport.close()

    async def start_run(self, kind, body, *, connection_override=None):
        if self._closing:
            raise AppError('backend_closing',status=409)
        if kind != "detection":
            raise AppError("collection_moved_to_cli")
        if self.updates.busy():
            raise AppError('update_waiting', status=409)
        connection, key = connection_override or self.connection(body)
        if not isinstance(key, str) or not key:
            raise AppError("credential_required")
        identity = body.get("resume_id") or uuid.uuid4().hex
        if identity in self.active:
            raise AppError("session_already_running", status=409)
        if body.get("resume_id"):
            saved = self.store.session(identity)
            if not saved or saved["kind"] != kind:
                raise AppError("resume_not_found")
            config = saved["config"]
            if connection.get("base_url") != config["base_url"]:
                raise AppError("frozen_configuration_mismatch")
            source = config["package"]
            self.catalog.check_withdrawal(source)
        else:
            config = {**connection, "runtime": body.get("runtime", {})}
            config["sample_ratio"] = .6
            config["version"] = __version__
            config["site_group"] = body.get("site_group", "")
            source = self.catalog.get(body.get("package_id"), body.get("package_version"))
            config["benchmark_publisher"] = next(item["publisher"] for item in self.catalog.local() if item["id"] == source["id"] and item["version"] == source["version"])
            config.update({"tier": body.get("tier", "low"), "claimed_model": body.get("claimed_model")})
            if config['tier'] not in source['tiers']:
                raise AppError('invalid_detection_configuration')
            config['runtime'] = detection_options(body.get('runtime', {}), sum(source['tiers'][config['tier']]['counts'].values()))
            if not config.get("request_model"):
                config["request_model"] = body.get("claimed_model")
        options = runtime_options(config.get("runtime", {}))
        sender = AsyncTransport([key], timeout=options["timeout"], concurrency=options["workers"], gates=self.rate_gates)
        try:
            runner = DetectorSession(
                self.store, identity, source, config, key, transport=sender)
        except Exception:
            await sender.close()
            raise
        task = asyncio.create_task(runner.run())
        self.active[identity] = (runner, task)
        task.add_done_callback(lambda _future: self.active.pop(identity, None))
        return identity

    async def scheduled_run(self, body):
        endpoint = body["endpoint_snapshot"]
        key = self.presets.vault.load(endpoint["credential_ref"])
        package = self.catalog.get(body["package_id"], body["package_version"])
        if package["content_sha256"] != body["package_sha256"]:
            raise AppError("scheduled_package_changed")
        connection = {"base_url": endpoint["base_url"], "allow_insecure": endpoint["allow_insecure"],
                      "request_model": body["request_model"]}
        identity = await self.start_run("detection", body, connection_override=(connection, key))
        await self.active[identity][1]
        return identity

    async def start_schedule(self, body):
        if self.schedule.task and not self.schedule.task.done():
            raise AppError("schedule_already_active", status=409)
        detection = body.get("detection", {})
        if not isinstance(detection, dict) or detection.get("key"):
            raise AppError("schedule_requires_saved_endpoint")
        endpoint = self.store.document("endpoint", detection.get("endpoint_id", ""))
        if not endpoint or not endpoint.get("credential_ref"):
            raise AppError("schedule_requires_saved_endpoint")
        package = self.catalog.get(detection.get("package_id"), detection.get("package_version"))
        if package["mode"] != endpoint["mode"]:
            raise AppError("endpoint_mode_mismatch")
        if detection.get("claimed_model") not in {model["id"] for model in package["models"]} or detection.get("tier", "low") not in package["tiers"]:
            raise AppError("invalid_detection_configuration")
        previous = self.store.document("schedule", "active")
        runtime = detection_options(detection.get("runtime", {}), sum(package['tiers'][detection.get('tier', 'low')]['counts'].values()))
        site_group = normalize_site_group(detection.get("site_group", ""))
        schedule_key = self.presets.vault.load(endpoint["credential_ref"])
        SecretGuard([schedule_key]).check(site_group, code="credential_in_configuration")
        reference = uuid.uuid4().hex
        self.presets.vault.save(reference, schedule_key)
        endpoint = {**endpoint, "credential_ref": reference, "schedule_owned": True}
        frozen = {"package_id": package["id"], "package_version": package["version"],
                  "package_sha256": package["content_sha256"], "endpoint_snapshot": endpoint,
                  "claimed_model": detection.get("claimed_model"),
                  "site_group": site_group,
                  "request_model": detection.get("request_model") or endpoint["model"],
                  "tier": detection.get("tier", "low"), "runtime": runtime}
        try:
            await self.schedule.start({"detection": frozen,
                "interval_seconds": body.get("interval_seconds", 3600), "round_limit": body.get("round_limit")})
        except Exception:
            self.presets.vault.delete(reference)
            raise
        old_endpoint = (previous or {}).get("detection", {}).get("endpoint_snapshot", {})
        if old_endpoint.get("schedule_owned"):
            self.presets.vault.delete(old_endpoint["credential_ref"])

    async def stop_run(self, identity):
        if identity in self.active:
            self.active[identity][0].stop()
        return {"stopping": identity}

    async def delete_schedule(self):
        if self.schedule.task and not self.schedule.task.done():
            raise AppError("schedule_already_active", status=409)
        saved = self.store.document("schedule", "active")
        endpoint = (saved or {}).get("detection", {}).get("endpoint_snapshot", {})
        if endpoint.get("schedule_owned"):
            self.presets.vault.delete(endpoint["credential_ref"])
        self.store.delete_document("schedule", "active")
        return {"deleted": True}

    def status(self):
        return {"sessions": self.store.session_summaries(), "active": list(self.active),
                "schedule": self.schedule.status()}

    def snapshot(self):
        return {**self.status(), "version": __version__, "brand": "meow LLM Detector", "packages": self.catalog.local(),
                "endpoints": self.presets.list(), "defaults": self.store.document("settings", "default_benchmarks") or {}, "catalog": self.catalog.index()}

    async def shutdown(self):
        self.log_maintenance_task.cancel()
        await asyncio.gather(self.log_maintenance_task, return_exceptions=True)
        self.schedule.pause()
        for runner, _task in tuple(self.active.values()):
            runner.stop()
        if self.active:
            await asyncio.gather(*(task for _runner, task in tuple(self.active.values())), return_exceptions=True)
        if self.schedule.task:
            await asyncio.gather(self.schedule.task, return_exceptions=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.call(self.shutdown())
        finally:
            self._resources.close()

    def _stop_loop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()


class Handler(BaseHTTPRequestHandler):
    server_version = "meow"
    protocol_version = "HTTP/1.0"

    def log_message(self, *_args):
        pass

    def _send(self, value, status=200, content_type="application/json; charset=utf-8", bootstrap=False):
        body = value if isinstance(value, bytes) else canonical_json(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; font-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        if bootstrap:
            self.send_header("Set-Cookie", f"{self.server.cookie_name}={self.server.token}; HttpOnly; SameSite=Strict; Path=/")
        self.end_headers()
        self.wfile.write(body)

    def _check(self, authenticated=True):
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in allowed:
            raise AppError("invalid_host", status=403)
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in allowed}:
            raise AppError("invalid_origin", status=403)
        if authenticated:
            cookie = SimpleCookie()
            cookie.load(self.headers.get("Cookie", ""))
            candidate = self.headers.get("X-Meow-Token") or (cookie[self.server.cookie_name].value if self.server.cookie_name in cookie else "")
            if not secrets.compare_digest(candidate, self.server.token):
                raise AppError("session_token_required", status=403)

    def do_GET(self):
        try:
            path = urlsplit(self.path).path
            self._check(authenticated=path not in ASSETS and path != "/api/bootstrap")
            if path in ASSETS:
                name, content_type = ASSETS[path]
                self._send((WEB_ROOT / name).read_bytes(), content_type=content_type, bootstrap=path == "/")
            elif path == "/api/bootstrap":
                self._send({"token": self.server.token, "version": __version__, "locale": self.server.state.locale, "tier_defaults": DEFAULT_TIER_COUNTS}, bootstrap=True)
            elif path == "/api/snapshot":
                self._send(self.server.state.snapshot())
            elif path == "/api/status":
                self._send(self.server.state.status())
            elif path == '/api/program/update-status':
                self._send(self.server.state.updates.status())
            elif path == '/api/reports':
                args = parse_qs(urlsplit(self.path).query)
                query = args.get('q', [''])[0].strip()
                if len(query) > 2048:
                    raise AppError('invalid_query')
                cursor = args.get('before', [None])[0]
                if cursor is not None:
                    if not cursor.isdecimal():
                        raise AppError('invalid_cursor')
                    cursor = integer(int(cursor), 'before', 1, 2**63-1)
                self._send(self.server.state.store.search_reports(query, cursor))
            elif path == "/api/legacy-work":
                state = self.server.state
                self._send({kind: state.store.documents(kind) for kind in ("project","simulation_task","simulation_result")})
            elif path == "/api/agent-kit":
                import io, zipfile
                output = io.BytesIO()
                with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                    for entry in KIT.glob("*.md"):
                        archive.writestr(entry.name, entry.read_bytes())
                self._send(output.getvalue(), content_type="application/zip")
            elif path.startswith("/api/retention/"):
                identity = path.removeprefix("/api/retention/")
                if not self.server.state.store.session(identity):
                    raise AppError("session_not_found", status=404)
                after = integer(int(parse_qs(urlsplit(self.path).query).get("after", ["0"])[0]), "after", 0, 2 ** 63 - 1)
                self._send(self.server.state.store.retained_exchanges(identity, after))
            elif path.startswith("/api/report/"):
                identity = path.removeprefix("/api/report/")
                report = self.server.state.store.report(identity)
                if identity in self.server.state.active:
                    report = self.server.state.active[identity][0].report()
                if not report:
                    raise AppError("report_not_found", status=404)
                self._send(report)
            elif path.startswith("/api/progress/"):
                try:
                    self._send(self.server.state.store.progress(path.removeprefix("/api/progress/")))
                except KeyError:
                    raise AppError("session_not_found", status=404)
            else:
                raise AppError("not_found", status=404)
        except AppError as exc:
            self._send({"error": exc.public()}, exc.status)
        except RequestError as exc:
            self._send({"error": exc.public()}, 502)
        except (OSError, ValueError):
            self._send({"error": {"code": "not_found"}}, 404)

    def do_POST(self):
        try:
            raw_length = self.headers.get("Content-Length", "")
            if not raw_length.isascii() or not raw_length.isdecimal() or not 0 < int(raw_length) <= MAX_PACKAGE_BYTES:
                raise AppError("invalid_content_length", status=413)
            self.connection.settimeout(10)
            raw = self.rfile.read(int(raw_length))
            if len(raw) != int(raw_length):
                raise AppError("incomplete_request")
            self._check()
            body = strict_json_loads(raw)
            if not isinstance(body, dict):
                raise AppError("invalid_request")
            path = urlsplit(self.path).path
            state = self.server.state
            if state._closing:
                raise AppError('backend_closing',status=409)
            if path == "/api/run/start":
                result = {"session_id": state.call(state.start_run("detection", body))}
            elif path == '/api/models':
                result = state.call(state.models(body), timeout=35)
            elif path == "/api/run/stop":
                result = state.call(state.stop_run(body.get("session_id")))
            elif path == "/api/run/estimate":
                package = state.catalog.get(body.get("package_id"), body.get("package_version"))
                tier = body.get('tier', 'low')
                if tier not in package['tiers']:raise AppError('invalid_detection_configuration')
                runtime = detection_options(body.get('runtime', {}), sum(package['tiers'][tier]['counts'].values()))
                result = estimate_plan(package, tier, retry_budget=runtime['retry_budget'])
            elif path == "/api/endpoint/save":
                result = state.presets.save(body.get("preset"), body.get("key"))
            elif path == "/api/endpoint/delete":
                state.presets.delete(body.get("id"))
                result = {"deleted": True}
            elif path == "/api/agent-prompt":
                result = task_prompt(body, data_root=state.root, locale=state.locale)
            elif path == "/api/catalog/default":
                package = state.catalog.get(body.get("id"), body.get("version"))
                defaults = state.store.document("settings", "default_benchmarks") or {}
                defaults[package["mode"]] = {"id": package["id"], "version": package["version"]}
                state.store.put_document("settings", "default_benchmarks", defaults)
                result = defaults
            elif path == "/api/catalog/update":
                result = state.call(state.install_baselines(body.get('packages')), timeout=300)
            elif path == "/api/catalog/refresh":
                result = state.call(state.catalog.refresh(), timeout=70)
            elif path == "/api/program/check-update":
                result = state.call(state.updates.check(body.get("locale", state.locale)), timeout=40)
            elif path == '/api/program/install-update':
                if body.get('confirmed') is not True:
                    raise AppError('update_download_confirmation_required')
                result = state.call(state.updates.install(body.get('version'), body.get('locale', state.locale), state, self.server))
            elif path == '/api/program/exit':
                if body.get('confirmed') is not True:
                    raise AppError('exit_confirmation_required')
                state.call(state.prepare_exit())
                self._send({'stopping':True})
                threading.Thread(target=self.server.shutdown,daemon=True).start()
                return
            elif path == "/api/program/download-update":
                if body.get("confirmed") is not True:
                    raise AppError("update_download_confirmation_required")
                result = state.call(state.updates.download(body.get("version"), body.get("locale", state.locale)), timeout=310)
            elif path == "/api/catalog/install":
                result = state.call(state.catalog.install(body.get("id"), body.get("version")), timeout=40)
            elif path == "/api/package/import":
                result = state.catalog.install_local(body.get("package"))
            elif path == "/api/package/export":
                result = state.catalog.get(body.get("id"), body.get("version"))
            elif path == "/api/schedule/start":
                state.call(state.start_schedule(body))
                result = {"started": True}
            elif path == "/api/schedule/pause":
                state.loop.call_soon_threadsafe(state.schedule.pause)
                result = {"paused": True}
            elif path == "/api/schedule/delete":
                result = state.call(state.delete_schedule())
            else:
                raise AppError("not_found", status=404)
            self._send(result)
        except AppError as exc:
            self._send({"error": exc.public()}, exc.status)
        except RequestError as exc:
            self._send({"error": exc.public()}, 502)
        except (TimeoutError, FutureTimeout):
            self._send({"error": {"code": "operation_timeout"}}, 504)
        except Exception:
            self._send({"error": {"code": "operation_failed"}}, 500)


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, state):
        self.state, self.token = state, secrets.token_urlsafe(32)
        self._slots = threading.BoundedSemaphore(32)
        super().__init__(address, Handler)
        self.cookie_name = f"meow_session_{self.server_port}"

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            request.close()
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            request.settimeout(10)
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def server_close(self):
        super().server_close()
        self.state.close()


def create_server(*, port=0, runs_root=None, locale="zh-CN"):
    root = Path(runs_root) if runs_root else Path(__file__).resolve().parent.parent / "meow_runs"
    state = AppState(root, locale)
    try:
        return AppServer(("127.0.0.1", port), state)
    except Exception:
        state.close()
        raise
