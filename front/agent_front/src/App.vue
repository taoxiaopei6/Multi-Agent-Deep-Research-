<script setup lang="ts">
import { nextTick, ref } from 'vue'

type StreamEvent = {
  type: 'status' | 'phase' | 'route' | 'final' | 'error' | 'evidence' | 'iteration' | 'trace' | 'artifact'
  message?: string
  final?: string
  node?: string
  round?: number
  trace?: {
    node: string
    status: string
    timestamp: string
    summary?: string
    output_summary?: Record<string, unknown>
    metrics?: {
      duration_ms?: number
      tokens?: {
        prompt_tokens?: number
        completion_tokens?: number
        total_tokens?: number
      }
    }
  }
  artifact?: {
    query: string
    claims: Array<{
      claim_id: string
      claim: string
      confidence: string
      evidence_count: number
      supporting_evidence: Array<{
        source_id: string
        title: string
        url: string
        source_type: string
        reliability_score: number | null
        reliability_breakdown?: Record<string, number>
      }>
    }>
    evidence_pool_summary: {
      total: number
      high_confidence: number
      low_confidence: number
    }
    audit_flags: number
    total_claims: number
    total_evidence: number
  }
  evidence_pool?: Array<{
    source_id: string
    title: string
    reliability_score: number
    reliability_reason?: string
    url?: string
    reliability_breakdown?: Record<string, number>
  }>
}

type ChatMessage = {
  id: string
  role: 'user' | 'assistant' | 'status'
  content: string
}

const userId = ref('user01')
const threadId = ref('thread01')
const tenantId = ref('default_tenant')
const query = ref('')
const loading = ref(false)
const errorMessage = ref('')
const messageListRef = ref<HTMLElement | null>(null)
const composerRef = ref<HTMLTextAreaElement | null>(null)
const progressLogs = ref<string[]>([])
const sourceScores = ref<Record<string, number>>({})
const traceEvents = ref<Array<{
  node: string
  status: string
  timestamp: string
  summary?: string
  output_summary?: Record<string, unknown>
}>>([])
const showTracePanel = ref(true)
const researchArtifact = ref<StreamEvent['artifact'] | null>(null)
const starterPrompts = [
  {
    title: '深度调研',
    prompt:
      '请调研“企业知识库 Agent 平台”市场，按市场规模、主要竞品、收费模式三部分输出，并在每部分附上可追溯来源链接。',
  },
  {
    title: '方案对比',
    prompt:
      '我们要做多 Agent 研究助手，请对比“纯大模型直答”“RAG 单 Agent”“多 Agent 协作”三种方案，给出优缺点、适用场景与推荐结论。',
  },
  {
    title: '知识问答',
    prompt: '请解释这个项目里“意图分流”的作用，以及简单问题和复杂问题分别会走哪条链路。',
  },
  {
    title: '落地计划',
    prompt: '请把“上线一个可用的 DeepResearch MVP”拆成两周计划，按每天输出任务、验收标准和风险点。',
  },
]
const capabilityHighlights = [
  {
    title: '多智能体编排',
    desc: '自动完成规划、检索、证据裁判、分析与写作，减少手工研究路径。',
  },
  {
    title: '双源检索融合',
    desc: '网络信息与本地知识库并行召回，输出结论同时保留来源可追溯性。',
  },
  {
    title: '会话记忆增强',
    desc: '跨轮次继承用户偏好与历史任务，持续提升回答一致性和效率。',
  },
]
const landingMetrics = [
  { label: '执行模式', value: 'Quick + Deep' },
  { label: '检索来源', value: 'Web + Local' },
  { label: '输出风格', value: '结论 + 证据' },
]
const messages = ref<ChatMessage[]>([
  {
    id: `m-${Date.now()}`,
    role: 'assistant',
    content: '你好，我是 DeepResearch。你可以直接提问，我会根据意图自动走快速回答或完整研究链路。',
  },
])

const escapeHtml = (value: string): string =>
  value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')

const markdownToHtml = (markdown: string): string => {
  const codeBlocks: string[] = []
  let text = markdown.replace(/```([\s\S]*?)```/g, (_, block) => {
    const index = codeBlocks.length
    codeBlocks.push(`<pre><code>${escapeHtml(String(block).trim())}</code></pre>`)
    return `@@CODE_BLOCK_${index}@@`
  })
  const lines = text.split('\n')
  const out: string[] = []
  let inList = false
  const closeList = () => {
    if (inList) {
      out.push('</ul>')
      inList = false
    }
  }
  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (!line) {
      closeList()
      continue
    }
    if (line.startsWith('# ')) {
      closeList()
      out.push(`<h1>${escapeHtml(line.slice(2))}</h1>`)
      continue
    }
    if (line.startsWith('## ')) {
      closeList()
      out.push(`<h2>${escapeHtml(line.slice(3))}</h2>`)
      continue
    }
    if (line.startsWith('### ')) {
      closeList()
      out.push(`<h3>${escapeHtml(line.slice(4))}</h3>`)
      continue
    }
    if (line.startsWith('- ') || line.startsWith('* ')) {
      if (!inList) {
        out.push('<ul>')
        inList = true
      }
      out.push(`<li>${escapeHtml(line.slice(2))}</li>`)
      continue
    }
    closeList()
    out.push(`<p>${escapeHtml(line)}</p>`)
  }
  closeList()
  let html = out.join('')
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>')
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>')
  html = html.replace(/\[([^[\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
  html = html.replace(/@@CODE_BLOCK_(\d+)@@/g, (_, idx) => codeBlocks[Number(idx)] || '')
  return html
}

const getBreakdownTooltip = (id: string): string => {
  const bdPrefix = `${id}_breakdown_`
  const authority = sourceScores.value[`${id}_detail`]
  if (authority === undefined) return ''
  const score = sourceScores.value[id]
  if (score === undefined) return ''
  // Read from the breakdown we stored during evidence events
  const stored = sourceScores.value[`${id}_bd`]
  if (stored && typeof stored === 'string') {
    return `Evidence Confidence: ${(score * 100).toFixed(0)}%\n${stored}`
  }
  return `Evidence Confidence: ${(score * 100).toFixed(0)}%`
}

const addReliabilityBadges = (html: string): string => {
  if (Object.keys(sourceScores.value).length === 0) return html
  return html.replace(/\[([^\]]+)\]/g, (match, id) => {
    const score = sourceScores.value[id]
    if (score === undefined) return match
    const cls = score >= 0.8 ? 'high' : score >= 0.6 ? 'med' : 'low'
    const label = score >= 0.8 ? 'High' : score >= 0.6 ? 'Med' : 'Low'
    return `<span class="rel-${cls}" title="Evidence Confidence: ${(score * 100).toFixed(0)}%">${match} <sup class="conf-badge ${cls}">${label}</sup></span>`
  })
}

const formatTraceStats = (stats: Record<string, unknown>): string => {
  return Object.entries(stats)
    .filter(([, v]) => v !== null && v !== undefined && v !== false)
    .map(([k, v]) => `${k}: ${v}`)
    .join(' | ')
}

const reliabilityClass = (score: number | null): string => {
  if (score === null) return 'unknown'
  return score >= 0.8 ? 'high' : score >= 0.6 ? 'med' : 'low'
}

const formatScore = (score: number | null): string => {
  if (score === null) return '--'
  return (score * 100).toFixed(0) + '%'
}

const formatDuration = (ms: number): string => {
  if (ms < 1000) return ms.toFixed(0) + 'ms'
  if (ms < 60000) return (ms / 1000).toFixed(1) + 's'
  return (ms / 60000).toFixed(1) + 'm'
}

const renderMessageHtml = (message: ChatMessage) => addReliabilityBadges(markdownToHtml(message.content || ''))

const scrollToBottom = async () => {
  await nextTick()
  const el = messageListRef.value
  if (el) {
    el.scrollTop = el.scrollHeight
  }
}

const createNewChat = () => {
  messages.value = [
    {
      id: `m-${Date.now()}`,
      role: 'assistant',
      content: '已开始新会话。你可以继续提问。',
    },
  ]
  progressLogs.value = []
  traceEvents.value = []
  researchArtifact.value = null
  errorMessage.value = ''
  query.value = ''
}

const usePrompt = async (prompt: string) => {
  query.value = prompt
  errorMessage.value = ''
  await nextTick()
  composerRef.value?.focus()
}

const applyStarterByIndex = (index: number) => {
  const target = starterPrompts[index]
  if (!target) return
  usePrompt(target.prompt)
}

const pushProgress = (message: string) => {
  const msg = message.trim()
  if (!msg) return
  const last = progressLogs.value[progressLogs.value.length - 1]
  if (last === msg) return
  progressLogs.value.push(msg)
  if (progressLogs.value.length > 6) {
    progressLogs.value = progressLogs.value.slice(-6)
  }
}

const runResearch = async () => {
  const userText = query.value.trim()
  if (!userText || loading.value) return
  loading.value = true
  errorMessage.value = ''
  progressLogs.value = []
  query.value = ''
  messages.value.push({ id: `u-${Date.now()}`, role: 'user', content: userText })
  const statusId = `s-${Date.now()}`
  messages.value.push({ id: statusId, role: 'status', content: '正在初始化执行链路...' })
  const renderStatusText = () => {
    const statusMessage = messages.value.find((item) => item.id === statusId)
    if (!statusMessage) return
    const latest = progressLogs.value.slice(-8)
    statusMessage.content = ['正在处理中...', ...latest].map((line) => `- ${line}`).join('\n')
  }
  renderStatusText()
  await scrollToBottom()
  try {
    const response = await fetch('/api/v1/research/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: userText,
        user_id: userId.value.trim() || 'default_user',
        thread_id: threadId.value.trim() || 'default_thread',
        tenant_id: tenantId.value.trim() || 'default_tenant',
      }),
    })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || `请求失败: ${response.status}`)
    }
    if (!response.body) {
      throw new Error('流式响应不可用')
    }
    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const parts = buffer.split('\n\n')
      buffer = parts.pop() || ''
      for (const part of parts) {
        if (!part.startsWith('data: ')) continue
        const jsonText = part.slice(6).trim()
        if (!jsonText) continue
        const event = JSON.parse(jsonText) as StreamEvent
        if (event.type === 'status' || event.type === 'phase' || event.type === 'route') {
          const prefix = event.type === 'phase' && event.node ? `[${event.node}] ` : ''
          pushProgress(`${prefix}${event.message || ''}`)
          renderStatusText()
        }
        if (event.type === 'trace' && event.trace) {
          traceEvents.value.push(event.trace)
          if (event.trace.summary) {
            pushProgress(event.trace.summary)
          }
          renderStatusText()
        }
        if (event.type === 'iteration') {
          pushProgress(`正在进行第 ${event.round} 轮补搜`)
          renderStatusText()
        }
        if (event.type === 'evidence') {
          if (event.evidence_pool) {
            for (const item of event.evidence_pool) {
              sourceScores.value[item.source_id] = item.reliability_score
              // Store breakdown as a string for tooltip display
              if (item.reliability_breakdown) {
                const bd = item.reliability_breakdown
                const breakdownStr = Object.entries(bd)
                  .filter(([, v]) => typeof v === 'number')
                  .map(([k, v]) => `${k}: ${(v * 100).toFixed(0)}%`)
                  .join(' | ')
                sourceScores.value[`${item.source_id}_bd`] = breakdownStr as never
              }
            }
          }
          pushProgress(`证据裁判完成，收集到 ${event.evidence_pool?.length || 0} 条证据`)
          renderStatusText()
        }
        if (event.type === 'final') {
          messages.value = messages.value.filter((item) => item.id !== statusId)
          messages.value.push({
            id: `a-${Date.now()}`,
            role: 'assistant',
            content: event.final || '已完成，但未返回正文。',
          })
        }
        if (event.type === 'artifact' && event.artifact) {
          researchArtifact.value = event.artifact
        }
        if (event.type === 'error') {
          throw new Error(event.message || '服务端执行异常')
        }
      }
      await scrollToBottom()
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '请求失败'
    messages.value = messages.value.filter((item) => item.id !== statusId)
    messages.value.push({
      id: `e-${Date.now()}`,
      role: 'assistant',
      content: `请求失败：${errorMessage.value}`,
    })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}
</script>

<template>
  <div class="chat-shell">
    <aside class="chat-sidebar">
      <div class="sidebar-brand">
        <p class="brand-badge">AI Copilot</p>
        <h1>DeepResearch</h1>
        <p class="brand-desc">多智能体研究工作台，支持快速回答与深度调研。</p>
      </div>
      <div class="sidebar-head">
        <button class="new-chat-btn" @click="createNewChat">新建会话</button>
      </div>
      <div class="quick-entry">
        <p class="section-title">推荐起手问题</p>
        <button
          v-for="item in starterPrompts.slice(0, 3)"
          :key="item.title"
          class="quick-entry-btn"
          @click="usePrompt(item.prompt)"
        >
          {{ item.title }}
        </button>
      </div>
      <div class="settings-group">
        <label>User ID</label>
        <input v-model="userId" class="sidebar-input" />
      </div>
      <div class="settings-group">
        <label>Thread ID</label>
        <input v-model="threadId" class="sidebar-input" />
      </div>
      <div class="settings-group">
        <label>Tenant ID</label>
        <input v-model="tenantId" class="sidebar-input" />
      </div>
      <p class="hint-text">当前会话记忆键：{{ userId }} / {{ threadId }}</p>
    </aside>

    <main class="chat-main">
      <header class="main-header">
        <div>
          <h2>DeepResearch Enterprise Workspace</h2>
          <p>面向业务团队的企业级智能研究台，支持从问题定义到结论落地的完整链路。</p>
        </div>
        <div class="header-tags">
          <span>Evidence-Driven</span>
          <span>Structured Output</span>
          <span>Memory-Powered</span>
        </div>
      </header>
      <div ref="messageListRef" class="message-list">
        <section v-if="messages.length <= 1" class="onboarding-panel">
          <div class="hero-panel">
            <p class="hero-badge">商业研究 · 策略分析 · 知识问答</p>
            <h3>第一步先讲清目标，再交给 DeepResearch 自动推进</h3>
            <p class="hero-desc">
              推荐提问结构：目标 + 背景约束 + 期望输出。系统会自动选择快速回答或深度研究链路。
            </p>
            <div class="hero-actions">
              <button class="hero-btn primary" @click="applyStarterByIndex(0)">快速开始调研</button>
              <button class="hero-btn" @click="applyStarterByIndex(1)">查看方案对比</button>
            </div>
            <div class="metric-grid">
              <article v-for="item in landingMetrics" :key="item.label">
                <p>{{ item.label }}</p>
                <strong>{{ item.value }}</strong>
              </article>
            </div>
          </div>
          <div class="capability-grid">
            <article v-for="item in capabilityHighlights" :key="item.title" class="capability-card">
              <h4>{{ item.title }}</h4>
              <p>{{ item.desc }}</p>
            </article>
          </div>
          <div class="guide-panel">
            <h4>提问指南</h4>
            <div class="guide-grid">
              <article>
                <h5>1. 说明目标</h5>
                <p>你要解决什么问题、面向谁、希望达到什么结果。</p>
              </article>
              <article>
                <h5>2. 提供上下文</h5>
                <p>给出已知信息、时间范围、数据口径、业务限制。</p>
              </article>
              <article>
                <h5>3. 指定输出</h5>
                <p>例如“表格输出”“附来源链接”“分点行动清单”。</p>
              </article>
            </div>
          </div>
          <div class="prompt-list">
            <button v-for="item in starterPrompts" :key="item.prompt" class="prompt-chip" @click="usePrompt(item.prompt)">
              {{ item.prompt }}
            </button>
          </div>
        </section>
        <div
          v-for="message in messages"
          :key="message.id"
          class="message-row"
          :class="`role-${message.role}`"
        >
          <div class="avatar">{{ message.role === 'user' ? '你' : message.role === 'status' ? '...' : 'AI' }}</div>
          <div class="bubble markdown-body" v-html="renderMessageHtml(message)"></div>
        </div>
      </div>
      <!-- Trace 时间线面板 -->
      <div v-if="traceEvents.length > 0" class="trace-panel" :class="{ collapsed: !showTracePanel }">
        <div class="trace-header" @click="showTracePanel = !showTracePanel">
          <span class="trace-title">Research Trace</span>
          <span class="trace-count">{{ traceEvents.length }} steps</span>
          <span class="trace-toggle">{{ showTracePanel ? '−' : '+' }}</span>
        </div>
        <div v-show="showTracePanel" class="trace-body">
          <div v-for="(ev, idx) in traceEvents" :key="idx" class="trace-step">
            <div class="trace-dot" :class="ev.status"></div>
            <div class="trace-content">
              <div class="trace-node-label">{{ ev.node }}</div>
              <div v-if="ev.summary" class="trace-summary">{{ ev.summary }}</div>
              <div class="trace-meta">
                <span v-if="ev.output_summary" class="trace-stats">{{ formatTraceStats(ev.output_summary) }}</span>
                <span v-if="ev.metrics" class="trace-metrics">
                  <span v-if="ev.metrics.duration_ms" class="trace-metric-item">
                    <span class="metric-icon">&#9201;</span>{{ formatDuration(ev.metrics.duration_ms) }}
                  </span>
                  <span v-if="ev.metrics.tokens?.total_tokens" class="trace-metric-item">
                    <span class="metric-icon">&#9881;</span>{{ ev.metrics.tokens.total_tokens }} tok
                  </span>
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <!-- Research Artifact 面板 -->
      <div v-if="researchArtifact" class="artifact-panel">
        <div class="artifact-header">
          <span class="artifact-title">Research Artifact</span>
          <span class="artifact-count">{{ researchArtifact.total_claims }} claims / {{ researchArtifact.total_evidence }} sources</span>
        </div>
        <div class="artifact-body">
          <div class="artifact-summary">
            <span class="artifact-badge high">{{ researchArtifact.evidence_pool_summary.high_confidence }} high</span>
            <span class="artifact-badge low">{{ researchArtifact.evidence_pool_summary.low_confidence }} low</span>
            <span v-if="researchArtifact.audit_flags > 0" class="artifact-badge warn">{{ researchArtifact.audit_flags }} flags</span>
          </div>
          <div v-for="c in researchArtifact.claims" :key="c.claim_id" class="artifact-claim">
            <div class="claim-header">
              <span class="claim-id">{{ c.claim_id }}</span>
              <span class="claim-conf" :class="c.confidence">{{ c.confidence }}</span>
              <span class="claim-ev-count">{{ c.evidence_count }} sources</span>
            </div>
            <div class="claim-text">{{ c.claim }}</div>
            <div v-if="c.supporting_evidence.length > 0" class="claim-evidence-list">
              <div v-for="ev in c.supporting_evidence" :key="ev.source_id" class="claim-evidence-item">
                <div class="ev-marker" :class="reliabilityClass(ev.reliability_score)"></div>
                <span class="ev-title">{{ ev.title }}</span>
                <span class="ev-score">{{ formatScore(ev.reliability_score) }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="composer">
        <textarea
          v-model="query"
          ref="composerRef"
          class="composer-input"
          :disabled="loading"
          placeholder="输入你的问题，回车发送（Shift + Enter 换行）"
          @keydown.enter.exact.prevent="runResearch"
        />
        <button class="send-btn" :disabled="loading || !query.trim()" @click="runResearch">
          {{ loading ? '处理中...' : '发送' }}
        </button>
      </div>
      <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
    </main>
  </div>
</template>
