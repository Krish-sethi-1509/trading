const configuredBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
export const API_BASE = configuredBase.replace(/\/$/, '')

export async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      Accept: 'application/json',
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...options.headers,
    },
  })

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    const detail = body?.detail || body?.message || `Request failed (${response.status})`
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return response.json()
}

export function formatTimestamp(value, options = {}) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  const formatOptions = {}
  if (options.dateStyle !== 'none') formatOptions.dateStyle = options.dateStyle || 'medium'
  if (options.timeStyle !== 'none') formatOptions.timeStyle = options.timeStyle || 'short'
  return new Intl.DateTimeFormat(undefined, formatOptions).format(date)
}
