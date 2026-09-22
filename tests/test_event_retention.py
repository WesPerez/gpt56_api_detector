import asyncio
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from gpt56_vnext.server import AppState
from gpt56_vnext.store import SQLiteStateStore


class EventRetentionTests(unittest.TestCase):
    def test_oldest_first_active_runs_and_frozen_reports_survive(self):
        with tempfile.TemporaryDirectory() as folder, SQLiteStateStore(Path(folder) / 'state.sqlite3') as store:
            for identity, status in [('finished', 'complete'), ('active', 'running')]:
                store.create_session(session_id=identity, kind='detection', status=status,
                                     config={'synthetic': True}, config_hash=identity, official=False)
            for offset in range(6):
                identity = store.append_event('finished', 'synthetic')
                store._write(lambda connection, identity=identity, offset=offset: connection.execute(
                    'UPDATE events SET created_at=? WHERE event_id=?',
                    (f'2026-09-22T00:00:0{offset}+00:00', identity)))
            active = store.append_event('active', 'synthetic')
            store._write(lambda connection: connection.execute(
                'UPDATE events SET created_at=? WHERE event_id=?', ('2026-09-01T00:00:00+00:00', active)))
            report = {'synthetic_result': True, 'events': store.events('finished')}
            store.save_report('finished', report)
            before = store.session('finished')
            now = datetime(2026, 9, 22, 1, tzinfo=timezone.utc)
            # The protected active row may temporarily exceed the ordinary cap.
            self.assertEqual(store.prune_events(now=now, max_events=3, batch_size=2), 2)
            self.assertEqual(store.prune_events(now=now, max_events=3), 1)
            self.assertEqual([row['time'] for row in store.events('finished')],
                             [f'2026-09-22T00:00:0{n}+00:00' for n in [3, 4, 5]])
            self.assertEqual(len(store.events('active')), 1)
            self.assertEqual(store.report('finished'), report)
            self.assertEqual(store.session('finished'), before)
            later = datetime(2026, 9, 26, tzinfo=timezone.utc)
            self.assertEqual(store.prune_events(now=later), 3)
            self.assertEqual(len(store.events('active')), 1)

    def test_current_boundary_is_retained(self):
        with tempfile.TemporaryDirectory() as folder, SQLiteStateStore(Path(folder) / 'state.sqlite3') as store:
            store.create_session(session_id='session', kind='detection', status='complete',
                                 config={}, config_hash='synthetic', official=False)
            for at in ['2026-09-18T23:59:59+00:00', '2026-09-19T00:00:00+00:00']:
                identity = store.append_event('session', 'synthetic')
                store._write(lambda connection, identity=identity, at=at: connection.execute(
                    'UPDATE events SET created_at=? WHERE event_id=?', (at, identity)))
            self.assertEqual(store.prune_events(now=datetime(2026, 9, 22, tzinfo=timezone.utc)), 1)
            self.assertEqual(store.events('session')[0]['time'], '2026-09-19T00:00:00+00:00')


class EventMaintenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_loop_stays_responsive_and_shutdown_waits_for_write(self):
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def prune():
            started.set()
            release.wait(5)
            finished.set()
            return 0

        state = SimpleNamespace(store=SimpleNamespace(prune_events=prune))
        task = asyncio.create_task(AppState._maintain_event_logs(state))
        try:
            self.assertTrue(await asyncio.wait_for(asyncio.to_thread(started.wait, 2), 3))
            self.assertFalse(finished.is_set())
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            self.assertTrue(finished.is_set())
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
