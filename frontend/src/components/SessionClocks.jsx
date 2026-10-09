import { useEffect, useState } from 'react'
import { DateTime } from 'luxon'

const MARKETS = [
  { name: 'Sydney', zone: 'Australia/Sydney', open: 7, close: 16, accent: 'sky' },
  { name: 'Tokyo', zone: 'Asia/Tokyo', open: 9, close: 18, accent: 'violet' },
  { name: 'London', zone: 'Europe/London', open: 8, close: 17, accent: 'amber' },
  { name: 'New York', zone: 'America/New_York', open: 8, close: 17, accent: 'emerald' },
]

function isOpen(now, market) {
  const local = now.setZone(market.zone)
  return local.weekday <= 5 && local.hour >= market.open && local.hour < market.close
}

export default function SessionClocks() {
  const [now, setNow] = useState(() => DateTime.now())
  const [localZone, setLocalZone] = useState('UTC')

  useEffect(() => {
    setLocalZone(Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC')
    const timer = window.setInterval(() => setNow(DateTime.now()), 1000)
    return () => window.clearInterval(timer)
  }, [])

  const utc = now.toUTC()
  const overlap = utc.weekday <= 5 && utc.hour >= 12 && utc.hour < 16

  return (
    <section className="panel clocks-panel">
      <div className="panel-heading clocks-heading">
        <div>
          <div className="eyebrow">GLOBAL MARKET HOURS</div>
          <h2>Session clocks</h2>
        </div>
        <div className="local-timezone">Local zone <strong>{localZone.replaceAll('_', ' ')}</strong></div>
      </div>
      <div className="clock-grid">
        {MARKETS.map((market) => {
          const local = now.setZone(market.zone)
          const open = isOpen(now, market)
          return (
            <div className={`clock-card accent-${market.accent}`} key={market.zone}>
              <div className="clock-card-top">
                <span className="clock-market">{market.name}</span>
                <span className={`market-status ${open ? 'is-open' : ''}`}><i />{open ? 'OPEN' : 'CLOSED'}</span>
              </div>
              <div className="clock-time">{local.toFormat('HH:mm:ss')}</div>
              <div className="clock-date">{local.toFormat('ccc, dd LLL')} <span>{local.toFormat('ZZZZ')}</span></div>
            </div>
          )
        })}
      </div>
      <div className={`overlap-banner ${overlap ? 'overlap-active' : ''}`}>
        <div className="overlap-icon">↗</div>
        <div className="overlap-copy">
          <strong>London / New York overlap</strong>
          <span>12:00–16:00 UTC · Peak liquidity window</span>
        </div>
        <span className={`overlap-state ${overlap ? 'active' : ''}`}>{overlap ? 'ACTIVE NOW' : 'INACTIVE'}</span>
      </div>
    </section>
  )
}
