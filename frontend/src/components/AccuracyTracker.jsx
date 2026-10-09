import { useCallback, useEffect, useState } from 'react'
import { apiRequest, formatTimestamp } from '../lib/api'

function directionClass(value) {
  return String(value || '').toLowerCase()
}

export default function AccuracyTracker() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  const load = useCallback(async (signal) => {
    try {
      const result = await apiRequest('/accuracy-log?limit=8', { signal })
      setData(result)
      setError('')
    } catch (requestError) {
      if (requestError.name !== 'AbortError') setError(requestError.message)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    load(controller.signal)
    const timer = window.setInterval(() => load(controller.signal), 60_000)
    return () => {
      controller.abort()
      window.clearInterval(timer)
    }
  }, [load])

  const accuracy = data?.accuracy_percent
  const recent = data?.recent || []

  return (
    <section className="panel accuracy-panel">
      <div className="panel-heading accuracy-heading">
        <div>
          <div className="eyebrow">LIVE PERFORMANCE</div>
          <h2>Accuracy tracker</h2>
        </div>
        <span className="updated-pill"><i /> AUTO-UPDATED</span>
      </div>
      <div className="accuracy-summary">
        <div>
          <div className="accuracy-value">{accuracy == null ? '—' : `${Number(accuracy).toFixed(1)}%`}</div>
          <div className="accuracy-label">Cumulative directional accuracy</div>
        </div>
        <div className="accuracy-count"><strong>{data?.correct_predictions ?? 0}<span>/{data?.scored_predictions ?? 0}</span></strong><small>correct calls</small></div>
      </div>
      <div className="accuracy-track"><div style={{ width: `${Math.max(0, Math.min(100, Number(accuracy) || 0))}%` }} /></div>

      <div className="table-title-row"><h3>Recent predictions</h3><span>{data?.pending_predictions ?? 0} pending</span></div>
      {error && <div className="inline-error" role="alert">{error}</div>}
      <div className="table-scroll">
        <table>
          <thead><tr><th>TIME</th><th>MODEL</th><th>ACTUAL</th><th>RESULT</th></tr></thead>
          <tbody>
            {recent.length ? recent.map((row, index) => (
              <tr key={`${row.timestamp}-${index}`}>
                <td className="table-time">{formatTimestamp(row.timestamp, { dateStyle: 'none' })}</td>
                <td><span className={`direction-tag ${directionClass(row.predicted_direction)}`}>{row.predicted_direction}</span></td>
                <td>{row.actual_outcome ? <span className={`direction-tag ${directionClass(row.actual_outcome)}`}>{row.actual_outcome}</span> : <span className="pending-tag">PENDING</span>}</td>
                <td>{row.correct == null ? <span className="result-pending">—</span> : <span className={`result-tag ${row.correct ? 'correct' : 'incorrect'}`}>{row.correct ? '✓ HIT' : '× MISS'}</span>}</td>
              </tr>
            )) : (
              <tr><td colSpan="4" className="empty-table">{data ? 'No predictions logged yet.' : 'Loading model history…'}</td></tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="accuracy-footnote">Accuracy updates as each 4-hour prediction reaches its evaluation window.</div>
    </section>
  )
}
