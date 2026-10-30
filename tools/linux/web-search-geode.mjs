// Geode's web search for DeepSeek Harness (/etc/geode/web-search-geode.mjs, loaded by dsh-geode.yml). Geode itself
// searches: geode-search (SearXNG's metasearch rewritten in Rust, AGPL-3.0, geode/search) runs on the phone outside
// this Linux, asks the search engines as Chrome on Android does - no search API, no key, no account - and nothing of
// it stays in memory between searches. The harness reaches it on the loopback server it reaches the model on, with
// the same key (Agent.kt: /geode/search).
export const name = 'web-search-geode'
export const inject = ['web']

export function apply(ctx) {
  ctx.web.registerSearchProvider({
    id: 'geode',
    available: () => Boolean(process.env.GEODE_BASE_URL && process.env.GEODE_KEY),
    search: (request, signal) => search(request, signal),
  })
}

async function search(request, signal) {
  const url = new URL('/geode/search', process.env.GEODE_BASE_URL)
  url.searchParams.set('q', request.query)
  if (request.maxResults) url.searchParams.set('max', String(request.maxResults))
  const res = await fetch(url, { headers: { authorization: `Bearer ${process.env.GEODE_KEY}` }, signal })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.error || `search failed (${res.status})`)
  return result(body)
}

// SearXNG's merged, ranked results as the harness's sources; its instant answers (an infobox's summary among them)
// as the provider's answer text
function result(r) {
  const answers = (r.answers ?? []).filter(Boolean)
  const sources = (r.results ?? []).filter((x) => x?.url).map((x) => ({
    url: x.url,
    ...(x.title ? { title: x.title } : {}),
    ...(x.content ? { snippet: x.content } : {}),
    ...(x.publishedDate ? { publishedAt: x.publishedDate } : {}),
  }))
  return { ...(answers.length ? { content: answers.join('\n') } : {}), sources, truncated: false }
}
