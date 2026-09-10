import { useCallback, useEffect, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, Download, RefreshCw, Search } from 'lucide-react';
import { accessLogsPath } from '../auth-api';
import { EmptyState, ErrorBanner, LoadingState, SectionCard, StatusBadge, formatDateTime } from './shared-ui';

const EVENTS = ['ACCESS_GRANTED', 'SPOOF_ATTEMPT', 'UNKNOWN_FACE', 'AMBIGUOUS_FACE', 'USER_DISABLED', 'PAD_UNAVAILABLE', 'PAD_ERROR', 'ACCESS_DENIED'];
const initialFilters = { username: '', event_type: '', result: '', start_time: '', end_time: '' };

export default function AccessLogsPanel({ request }) {
  const [logs, setLogs] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [draft, setDraft] = useState(initialFilters);
  const [filters, setFilters] = useState(initialFilters);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState('');
  const pending = useRef(null);
  const pageSize = 20;

  const fetchLogs = useCallback(async () => {
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setLoading(true); setError('');
    try {
      const response = await request(accessLogsPath('/access-logs', {
        ...filters, start_time: filters.start_time ? `${filters.start_time} 00:00:00` : '',
        end_time: filters.end_time ? `${filters.end_time} 23:59:59` : '',
        limit: pageSize, offset: (page - 1) * pageSize,
      }), { signal: controller.signal });
      const data = await response.json();
      if (!controller.signal.aborted) { setLogs(Array.isArray(data.items) ? data.items : []); setTotal(Number(data.total) || 0); }
    } catch (err) {
      if (!controller.signal.aborted) { setLogs([]); setTotal(0); setError(err.message); }
    } finally { if (!controller.signal.aborted) setLoading(false); }
  }, [filters, page, request]);

  useEffect(() => { fetchLogs(); return () => pending.current?.abort(); }, [fetchLogs]);

  const applyFilters = event => { event.preventDefault(); setPage(1); setFilters(draft); };
  const exportLogs = async () => {
    if (exporting) return;
    setExporting(true); setError('');
    try {
      const response = await request(accessLogsPath('/access-logs/export', {
        ...filters, start_time: filters.start_time ? `${filters.start_time} 00:00:00` : '',
        end_time: filters.end_time ? `${filters.end_time} 23:59:59` : '',
      }));
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement('a');
      link.href = url; link.download = `vshield-access-logs-${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(link);
      try { link.click(); } finally { link.remove(); URL.revokeObjectURL(url); }
    } catch (err) { setError(err.message); } finally { setExporting(false); }
  };

  const pages = Math.max(1, Math.ceil(total / pageSize));
  return <SectionCard id="logs" title="Access logs" description="Audit trail for biometric checks, liveness decisions and access results."
    action={<button className="button button--secondary button--small" onClick={fetchLogs} disabled={loading}><RefreshCw className={loading ? 'spin' : ''} size={16} />Refresh</button>}>
    <form className="filter-bar" onSubmit={applyFilters}>
      <label><span>Username</span><input value={draft.username} onChange={e => setDraft({ ...draft, username: e.target.value })} placeholder="Search account" /></label>
      <label><span>Event type</span><select value={draft.event_type} onChange={e => setDraft({ ...draft, event_type: e.target.value })}>
        <option value="">All events</option>{EVENTS.map(event => <option key={event}>{event}</option>)}
      </select></label>
      <label><span>Result</span><select value={draft.result} onChange={e => setDraft({ ...draft, result: e.target.value })}><option value="">All results</option><option>GRANTED</option><option>DENIED</option></select></label>
      <label><span>From</span><input type="date" value={draft.start_time} onChange={e => setDraft({ ...draft, start_time: e.target.value })} /></label>
      <label><span>To</span><input type="date" value={draft.end_time} onChange={e => setDraft({ ...draft, end_time: e.target.value })} /></label>
      <button className="button button--primary button--small" disabled={loading}><Search size={16} />Apply</button>
      <button type="button" className="button button--secondary button--small" onClick={exportLogs} disabled={exporting || !total}><Download size={16} />{exporting ? 'Exporting…' : 'Export CSV'}</button>
    </form>
    <ErrorBanner message={error} />
    {loading ? <LoadingState label="Loading security events…" /> : logs.length === 0 ?
      <EmptyState title="No matching events" message="Adjust the filters or perform an authentication attempt." /> :
      <div className="table-wrap"><table className="data-table"><thead><tr><th>Time</th><th>Identity</th><th>Event</th><th>Result</th><th>Recognition distance</th><th>Spoof score</th><th>Source</th><th>Reason</th></tr></thead>
        <tbody>{logs.map(log => <tr key={log.id}>
          <td className="nowrap">{formatDateTime(log.timestamp)}</td>
          <td><strong>{log.username || 'Unknown visitor'}</strong>{log.name && <small>{log.name}</small>}</td>
          <td><StatusBadge value={log.event_type} /></td><td><StatusBadge value={log.result} /></td>
          <td className="metric">{Number.isFinite(Number(log.recognition_distance)) && log.recognition_distance !== null ? Number(log.recognition_distance).toFixed(3) : '—'}</td>
          <td className="metric">{Number.isFinite(Number(log.spoof_score)) && log.spoof_score !== null ? Number(log.spoof_score).toFixed(3) : '—'}</td>
          <td>{log.source || '—'}</td><td>{log.reason_code || '—'}</td>
        </tr>)}</tbody></table></div>}
    <footer className="pagination"><span>{total ? `${(page - 1) * pageSize + 1}–${Math.min(page * pageSize, total)} of ${total}` : '0 events'}</span>
      <div><button className="icon-btn" aria-label="Previous page" onClick={() => setPage(value => value - 1)} disabled={page === 1 || loading}><ChevronLeft size={18} /></button><span>Page {page} / {pages}</span><button className="icon-btn" aria-label="Next page" onClick={() => setPage(value => value + 1)} disabled={page === pages || loading}><ChevronRight size={18} /></button></div></footer>
  </SectionCard>;
}
