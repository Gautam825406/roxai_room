import { useState, type FormEvent } from 'react'
import {
  BarVisualizer,
  DisconnectButton,
  LiveKitRoom,
  RoomAudioRenderer,
  StartAudio,
  useChat,
  useConnectionQualityIndicator,
  useConnectionState,
  useParticipants,
  useSpeakingParticipants,
  useTrackToggle,
  useTracks,
} from '@livekit/components-react'
import { ConnectionQuality, ConnectionState, Track } from 'livekit-client'
import type { ReceivedChatMessage, TrackReference } from '@livekit/components-react'
import type { LocalParticipant, RemoteParticipant } from 'livekit-client'
import type { TokenResponse } from './api'

interface RoomViewProps {
  session: TokenResponse
  onLeave: () => void
}

type Role = 'dost' | 'sathi' | 'human'

export function RoomView({ session, onLeave }: RoomViewProps) {
  return (
    <LiveKitRoom
      serverUrl={session.url}
      token={session.token}
      connect
      audio
      video={false}
      onDisconnected={onLeave}
      className="console"
    >
      <RoomAudioRenderer />
      <ConnectingOverlay />
      <div className="start-audio-banner">
        <StartAudio label="🔊 Click to enable sound" />
      </div>
      <Sidebar session={session} />
      <div className="console-main">
        <TopBar session={session} />
        <div className="workspace">
          <VoiceMesh session={session} />
          <TranscriptPanel session={session} />
        </div>
      </div>
    </LiveKitRoom>
  )
}

function ConnectingOverlay() {
  const state = useConnectionState()
  if (state === ConnectionState.Connected) return null
  const label = state === ConnectionState.Reconnecting ? 'Reconnecting…' : 'Connecting…'
  return (
    <div className="connecting-overlay">
      <div className="connecting-spinner" aria-hidden />
      <span>{label}</span>
    </div>
  )
}

function roleOf(name: string, session: TokenResponse): Role {
  if (name === session.bots[0]) return 'dost'
  if (name === session.bots[1]) return 'sathi'
  return 'human'
}

function initials(name: string): string {
  const parts = name.trim().split(/\s+/)
  const letters = parts.length > 1 ? parts[0][0] + parts[parts.length - 1][0] : parts[0]?.slice(0, 2)
  return (letters || '?').toUpperCase()
}

/* ---------------------------------- Sidebar ---------------------------------- */

function Sidebar({ session }: { session: TokenResponse }) {
  const participants = useParticipants()
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">🎙️</div>
        <div className="brand-text">
          <span className="brand-name">ROXROOM</span>
          <span className="brand-sub">VOICE ENGINE</span>
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">
          <span>Room</span>
        </div>
        <div className="channel-item active">
          <span className="channel-icon">👥</span>
          <span className="channel-name">{session.room}</span>
          <span className="live-dot" aria-hidden />
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">
          <span>Participants</span>
          <span className="mono sidebar-count">{participants.length}</span>
        </div>
        <div className="sidebar-people">
          {participants.map((p) => {
            const role = roleOf(p.name || '', session)
            return (
              <div key={p.identity} className={`sidebar-person role-${role}`}>
                <span className="sidebar-person-dot" aria-hidden />
                <span className="sidebar-person-name">
                  {p.isLocal ? 'You' : p.name || p.identity}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      <div className="sidebar-footer">
        <span className="sidebar-footer-dot" aria-hidden />
        <div className="sidebar-footer-text">
          <span>LiveKit</span>
          <span className="mono sidebar-footer-url">{hostOf(session.url)}</span>
        </div>
      </div>
    </aside>
  )
}

function hostOf(url: string): string {
  try {
    return new URL(url).host
  } catch {
    return url
  }
}

/* ---------------------------------- Top bar ---------------------------------- */

function TopBar({ session }: { session: TokenResponse }) {
  const state = useConnectionState()
  const participants = useParticipants()
  const { toggle: toggleMic, enabled: micEnabled } = useTrackToggle({ source: Track.Source.Microphone })

  const stateLabel =
    state === ConnectionState.Connected
      ? 'CONNECTED'
      : state === ConnectionState.Reconnecting
        ? 'RECONNECTING'
        : 'CONNECTING'

  return (
    <header className="topbar">
      <div className="topbar-left">
        <span className="topbar-room">Room: {session.room}</span>
        <div className={`state-pill state-${state}`}>
          <span className="state-dot" aria-hidden />
          <span className="mono">{stateLabel}</span>
        </div>
        <div className="people-pill">
          <span className="mono">{participants.length}</span>
          <span className="people-pill-names">
            ({participants.map((p) => (p.isLocal ? 'You' : p.name || p.identity)).join(', ')})
          </span>
        </div>
      </div>
      <div className="topbar-right">
        <button
          type="button"
          className={`icon-btn ${micEnabled ? '' : 'icon-btn-off'}`}
          onClick={() => toggleMic()}
          title={micEnabled ? 'Mute microphone' : 'Unmute microphone'}
        >
          {micEnabled ? '🎤' : '🔇'}
        </button>
        <DisconnectButton className="icon-btn icon-btn-danger" title="Leave room">
          📞
        </DisconnectButton>
      </div>
    </header>
  )
}

/* ---------------------------------- Voice mesh ---------------------------------- */

function VoiceMesh({ session }: { session: TokenResponse }) {
  const participants = useParticipants()
  const micTracks = useTracks([Track.Source.Microphone], { onlySubscribed: false })
  const speaking = useSpeakingParticipants()
  const speakingIds = new Set(speaking.map((p) => p.identity))

  const trackFor = (identity: string): TrackReference | undefined =>
    micTracks.find((t) => t.participant.identity === identity)

  const ordered = [...participants].sort((a, b) => {
    const rank = (p: typeof a) => (roleOf(p.name || '', session) === 'human' ? 0 : 1)
    return rank(a) - rank(b)
  })

  return (
    <section className="stage-panel">
      <div className="panel-heading">
        <h2>
          <span className="panel-icon">🌐</span>Spatial Voice Mesh
        </h2>
        <span className="pill-muted mono">{participants.length} Active Peers</span>
      </div>
      <div className="mesh-grid">
        {ordered.map((p) => (
          <ParticipantTile
            key={p.identity}
            participant={p}
            role={roleOf(p.name || '', session)}
            track={trackFor(p.identity)}
            isSpeaking={speakingIds.has(p.identity)}
          />
        ))}
      </div>
    </section>
  )
}

const ROLE_META: Record<Role, { label: string; icon: string; sub: string }> = {
  dost: { label: 'Roxstar AI Dost', icon: '🤖', sub: 'Warm, direct Hinglish buddy' },
  sathi: { label: 'Roxstar AI Sathi', icon: '💫', sub: 'Patient, explains with examples' },
  human: { label: '', icon: '🎧', sub: 'Human' },
}

function ParticipantTile({
  participant,
  role,
  track,
  isSpeaking,
}: {
  participant: RemoteParticipant | LocalParticipant
  role: Role
  track: TrackReference | undefined
  isSpeaking: boolean
}) {
  const meta = ROLE_META[role]
  const name = participant.isLocal ? 'You' : participant.name || participant.identity
  const { quality } = useConnectionQualityIndicator({ participant })
  const isMicOff = role === 'human' && (!track || track.publication.isMuted)

  const status = isSpeaking ? 'SPEAKING' : isMicOff ? 'MUTED' : role === 'human' ? 'LISTENING' : 'IDLE'

  return (
    <div className={`tile role-${role} ${isSpeaking ? 'speaking' : ''}`}>
      <div className="tile-glow" aria-hidden />
      <div className="tile-top">
        <span className={`status-pill status-${status.toLowerCase()}`}>
          <span className="status-dot" aria-hidden />
          {status}
        </span>
        {role === 'human' ? (
          <span className="mono tile-quality" data-quality={quality}>
            {isMicOff ? '🔇' : '🎙️'} {qualityLabel(quality)}
          </span>
        ) : (
          <span className="role-tag">{role.toUpperCase()}</span>
        )}
      </div>

      <div className="tile-center">
        <div className="tile-avatar-wrap">
          <div className="tile-ring" aria-hidden />
          <div className="tile-avatar">{role === 'human' ? initials(name) : meta.icon}</div>
        </div>
        <div className="tile-bars">
          {track ? (
            <BarVisualizer track={track} barCount={9} options={{ minHeight: 15, maxHeight: 100 }} />
          ) : (
            <div className="tile-bars-placeholder" />
          )}
        </div>
      </div>

      <div className="tile-footer">
        <span className="tile-name">{name}</span>
        <span className="tile-sub">{role === 'human' ? 'Human participant' : meta.sub}</span>
      </div>
    </div>
  )
}

function qualityLabel(q: ConnectionQuality): string {
  switch (q) {
    case ConnectionQuality.Excellent:
      return 'Excellent'
    case ConnectionQuality.Good:
      return 'Good'
    case ConnectionQuality.Poor:
      return 'Poor'
    default:
      return '—'
  }
}

/* ---------------------------------- Transcript panel ---------------------------------- */

function TranscriptPanel({ session }: { session: TokenResponse }) {
  const { chatMessages, send, isSending } = useChat()
  const [draft, setDraft] = useState('')

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const text = draft.trim()
    if (!text) return
    setDraft('')
    await send(text)
  }

  return (
    <aside className="transcript-panel">
      <div className="panel-heading">
        <h2>
          <span className="panel-icon">💬</span>Live Transcript
        </h2>
      </div>
      <div className="transcript-feed">
        {chatMessages.length === 0 && (
          <p className="transcript-empty">Say something, or type below — replies show up here as they happen.</p>
        )}
        {chatMessages.map((m) => (
          <TranscriptEntry key={m.id} message={m} session={session} />
        ))}
      </div>
      <form className="transcript-input" onSubmit={handleSubmit}>
        <span className="transcript-input-icon">🎤</span>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Type a Hinglish message…"
        />
        <button type="submit" disabled={!draft.trim() || isSending}>
          Send →
        </button>
      </form>
    </aside>
  )
}

function TranscriptEntry({ message, session }: { message: ReceivedChatMessage; session: TokenResponse }) {
  const name = message.from?.isLocal ? 'You' : message.from?.name || message.from?.identity || 'Unknown'
  const role = roleOf(message.from?.name || '', session)
  const time = new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

  return (
    <div className="transcript-entry">
      <div className={`transcript-avatar role-${role}`}>
        {role === 'human' ? initials(name) : ROLE_META[role].icon}
      </div>
      <div className="transcript-body">
        <div className="transcript-meta">
          <span className={`transcript-name role-${role}`}>{name}</span>
          <span className="mono transcript-time">{time}</span>
        </div>
        <div className="transcript-bubble">{message.message}</div>
      </div>
    </div>
  )
}
