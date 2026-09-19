import { useState, type FormEvent } from 'react'
import { fetchToken, TokenRequestError, type TokenResponse } from './api'

interface JoinScreenProps {
  onJoined: (session: TokenResponse) => void
}

export function JoinScreen({ onJoined }: JoinScreenProps) {
  const [name, setName] = useState('')
  const [room, setRoom] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()
    if (!name.trim()) return

    setBusy(true)
    setError(null)
    try {
      const session = await fetchToken(name.trim(), room.trim() || undefined)
      onJoined(session)
    } catch (err) {
      setError(err instanceof TokenRequestError ? err.message : 'Failed to join the room.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="join-screen">
      <div className="join-card">
        <div className="join-logo" aria-hidden>
          🎙️
        </div>
        <h1>RoxRoom AI</h1>
        <p className="subtitle">Talk to Roxstar AI Dost &amp; Sathi in a shared voice room.</p>
        <form onSubmit={handleSubmit}>
          <label htmlFor="name">Your name</label>
          <input
            id="name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Priya"
            autoFocus
            required
          />

          <label htmlFor="room">Room (optional)</label>
          <input
            id="room"
            value={room}
            onChange={(e) => setRoom(e.target.value)}
            placeholder="roxroom-dev"
          />

          {error && <p className="error">{error}</p>}

          <button type="submit" disabled={busy || !name.trim()}>
            {busy ? 'Joining…' : 'Join room'}
          </button>
        </form>
      </div>
    </div>
  )
}
