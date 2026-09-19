# RoxRoom AI — frontend

A small React app for humans to join the RoxRoom AI voice room: enter a name, get a
LiveKit token from the token server, then talk (mic) and chat with Roxstar AI Dost /
Sathi from the browser instead of the CLI + meet.livekit.io flow.

## Run it

You need two things running alongside your existing bot process(es):

1. The token server (mints LiveKit join tokens over HTTP):

   ```powershell
   # from ../backend, with .venv set up per ../SETUP.md
   cd ../backend
   .venv\Scripts\python.exe -m pip install -e ".[server]"
   .venv\Scripts\python.exe -m roxroom.token_server
   ```

   Listens on `http://localhost:8787`. Uses the same `.env` (`LIVEKIT_URL`,
   `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `ROXROOM_ROOM_NAME`) as everything else.

2. This app:

   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

   Open the printed `http://localhost:5173` URL. `.env.local` points it at the token
   server (`VITE_TOKEN_SERVER_URL`, defaults to `http://localhost:8787`).

3. In another terminal, run the actual bot(s) so there's someone to talk to:

   ```powershell
   cd backend
   .venv\Scripts\python.exe -m roxroom.run_bot
   ```

Enter a name on the join screen, click **Join room**, allow microphone access. You'll
see Dost/Sathi (and any other humans) in the participant list, hear bot replies, and
can use the chat panel — it's the same `lk.chat` topic the Python backend already reads
and writes, so text sent here reaches the orchestrator exactly like typing in the
meet.livekit.io chat did.

## Structure

- `src/api.ts` — fetches a token from the token server.
- `src/JoinScreen.tsx` — name entry form.
- `src/RoomView.tsx` — connects to LiveKit (`@livekit/components-react`); renders the
  audio-reactive avatar "stage" for Dost/Sathi/you (`BarVisualizer` bound to each
  participant's live track, glowing when they're actually speaking), the participant
  list, chat panel, mic toggle, and leave button.
