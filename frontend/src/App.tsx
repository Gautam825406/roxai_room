import { useState } from 'react'
import '@livekit/components-styles'
import './App.css'
import { JoinScreen } from './JoinScreen'
import { RoomView } from './RoomView'
import type { TokenResponse } from './api'

function App() {
  const [session, setSession] = useState<TokenResponse | null>(null)

  if (!session) {
    return <JoinScreen onJoined={setSession} />
  }

  return <RoomView session={session} onLeave={() => setSession(null)} />
}

export default App
