import { useState } from 'react'
import { apiRequest, formatTimestamp } from '../lib/api'

const directionCopy = {
  UP: { label: 'UP', icon: '↗', subtitle: 'Bullish directional bias' },
  DOWN: { label: 'DOWN', icon: '↘', subtitle: 'Bearish directional bias' },
  NEUTRAL: { label: 'NEUTRAL', icon: '↔', subtitle: 'No strong directional edge' },
}

export default function PredictionPanel() {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function runPrediction() {
    setLoading(true)
    setError('')
    try {
      const result = await apiRequest('/predict', { method: 'POST' })
      setPrediction(result)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }

  const direction = prediction?.direction?.toUpperCase()
  const view = directionCopy[direction]
  const confidence = Math.max(0, Math.min(100, Number(prediction?.confidence || 0) * 100))

  return (
    <section className="panel prediction-panel">
      <div className="panel-heading">
        <div>
          <div className="eyebrow">MODEL SIGNAL · 4H HORIZON</div>
          <h2>Directional outlook</h2>
        </div>
        <span className="model-chip"><span /> XGBOOST</span>
      </div>

      {prediction && view ? (
        <div className={`prediction-result direction-${direction.toLowerCase()}`}>
          <div className="direction-line">
            <div className="direction-icon">{view.icon}</div>
            <div>
              <div className="direction-label">{view.label}</div>
              <div className="direction-subtitle">{view.subtitle}</div>
            </div>
          </div>
          <div className="confidence-row">
            <span>Model confidence</span>
            <strong>{confidence.toFixed(1)}<small>%</small></strong>
          </div>
          <div className="confidence-track"><div className="confidence-fill" style={{ width: `${confidence}%` }} /></div>
          <div className="prediction-timestamp">Generated {formatTimestamp(prediction.timestamp)}</div>
        </div>
      ) : (
        <div className="prediction-empty">
          <div className="empty-orbit">✳</div>
          <strong>{loading ? 'Evaluating latest features' : 'No prediction generated'}</strong>
          <span>{loading ? 'The model is scoring the current feature vector.' : 'Run the model to view its next 4-hour directional bias.'}</span>
        </div>
      )}

      {error && <div className="inline-error" role="alert">{error}</div>}
      <button className="primary-button prediction-button" type="button" onClick={runPrediction} disabled={loading}>
        {loading ? <><span className="button-spinner" /> Running model…</> : prediction ? 'Refresh prediction' : 'Generate 4-hour prediction'}
      </button>
      <div className="disclaimer"><span>ⓘ</span><strong>Decision Support System — Not Financial Advice</strong><p>Predictions are probabilistic research outputs, not guarantees of future performance.</p></div>
    </section>
  )
}
