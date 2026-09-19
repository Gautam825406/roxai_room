export interface TokenResponse {
  token: string
  url: string
  identity: string
  room: string
  bots: string[]
}

const TOKEN_SERVER_URL = import.meta.env.VITE_TOKEN_SERVER_URL ?? 'http://localhost:8787'

export class TokenRequestError extends Error {}

export async function fetchToken(name: string, room?: string): Promise<TokenResponse> {
  let response: Response
  try {
    response = await fetch(`${TOKEN_SERVER_URL}/api/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, room }),
    })
  } catch {
    throw new TokenRequestError(
      `Could not reach the token server at ${TOKEN_SERVER_URL}. Is it running ` +
        '(python -m roxroom.token_server)?',
    )
  }

  if (!response.ok) {
    const detail = await response.text()
    throw new TokenRequestError(`Token server returned ${response.status}: ${detail}`)
  }

  return (await response.json()) as TokenResponse
}
