package xyz.tifu.geode.agent

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Debug builds only: the agent's Linux driven from adb, inside the app's process (its SELinux domain, its W^X rule):
 *
 *     adb shell am broadcast -n xyz.tifu.geode/.agent.LinuxDebugReceiver --es op install --es file linux/rootfs.tar.gz
 *     adb shell am broadcast -n xyz.tifu.geode/.agent.LinuxDebugReceiver --es op run --es cmd 'python3 -V'
 *     adb shell am broadcast -n xyz.tifu.geode/.agent.LinuxDebugReceiver --es op attach --es file debug/image.png
 *     adb shell am broadcast -n xyz.tifu.geode/.agent.LinuxDebugReceiver --es op attach_audio --es file debug/voice.wav
 *     adb shell am broadcast -n xyz.tifu.geode/.agent.LinuxDebugReceiver --es op attach_video --es file debug/video.mp4
 *
 * The result goes to files/linux/debug.log (adb shell run-as xyz.tifu.geode cat files/linux/debug.log).
 */
class LinuxDebugReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val app = context.applicationContext
        val op = intent.getStringExtra("op")
        val log = File(app.filesDir, "linux/debug.log").apply { parentFile?.mkdirs() }
        log.writeText("running $op\n")
        scope.launch {
            val linux = Linux(app)
            val t0 = System.currentTimeMillis()
            val text = runCatching {
                when (op) {
                    "install" -> {
                        val src = File(app.filesDir, intent.getStringExtra("file") ?: "linux/rootfs.tar.gz")
                        linux.install(src.inputStream(), "dev")
                        "installed"
                    }
                    "run" -> {
                        val ws = File(app.filesDir, "agents/" + (intent.getStringExtra("ws") ?: "debug"))
                        // kept alive and online with the phone locked (apt, pip reach the internet)
                        xyz.tifu.geode.engine.AnswerService.start(app)
                        val (code, out) = try {
                            linux.run(ws, intent.getStringExtra("cmd") ?: "uname -a", intent.getLongExtra("timeout", 600))
                        } finally {
                            xyz.tifu.geode.engine.AnswerService.stop(app)
                        }
                        "exit $code\n$out"
                    }
                    "agent" -> {
                        // the agent end to end, its events one a line (the live words left out)
                        val sb = StringBuilder()
                        // as the chat does: the app kept alive (adb: cmd deviceidle tempwhitelist lets the service
                        // start from here)
                        xyz.tifu.geode.engine.AnswerService.start(app)
                        val agent = (app as xyz.tifu.geode.GeodeApp).agent
                        val t1 = System.currentTimeMillis()
                        val stop = try {
                            agent.prompt(intent.getStringExtra("ws") ?: "debug",
                                intent.getStringExtra("session"), intent.getStringExtra("cmd") ?: "Say hi.") { e ->
                                if (e !is AgentEvent.Text && e !is AgentEvent.Thinking) synchronized(sb) {
                                    sb.appendLine("${System.currentTimeMillis() - t1} ms " + e.toString().take(600))
                                    log.writeText("running agent\n$sb")
                                }
                            }
                        } finally {
                            xyz.tifu.geode.engine.AnswerService.stop(app)
                        }
                        "stop $stop\n$sb"
                    }
                    // an image (a file in the app's own storage, adb-pushed) attached to the open chat's next question,
                    // as the photo picker would
                    "attach" -> {
                        val src = File(app.filesDir, intent.getStringExtra("file") ?: "debug/image.png")
                        val name = xyz.tifu.geode.vision.ImageStore(app).importChatBytes(src.readBytes())
                        xyz.tifu.geode.util.DebugHooks.attach.tryEmit(name)
                        "attached $name"
                    }
                    // a sound file (adb-pushed: any format Android decodes) attached as a clip, as the sound picker would
                    "attach_audio" -> {
                        val src = File(app.filesDir, intent.getStringExtra("file") ?: "debug/voice.wav")
                        val pcm = xyz.tifu.geode.media.AudioDecoder.decode(src)
                        val name = xyz.tifu.geode.media.AudioStore(app).save(pcm)
                        xyz.tifu.geode.util.DebugHooks.attachAudio.tryEmit(name)
                        "attached $name (${pcm.size} samples)"
                    }
                    // a video file (adb-pushed) attached to the open chat's next question, as the video picker would
                    "attach_video" -> {
                        val src = File(app.filesDir, intent.getStringExtra("file") ?: "debug/video.mp4")
                        xyz.tifu.geode.util.DebugHooks.attachVideo.tryEmit(android.net.Uri.fromFile(src))
                        "attaching ${src.name} (${src.length()} bytes)"
                    }
                    // an outside service's model saved and chosen for the chats (as the Models screen's dialog would),
                    // or the phone's own chosen again (provider_off)
                    "provider" -> {
                        val g = app as xyz.tifu.geode.GeodeApp
                        val p = xyz.tifu.geode.models.Provider(
                            id = intent.getStringExtra("id") ?: "pdebug",
                            name = intent.getStringExtra("name") ?: "Debug",
                            baseUrl = intent.getStringExtra("url") ?: "http://127.0.0.1:18080/v1",
                            protocol = xyz.tifu.geode.models.Protocol.valueOf(intent.getStringExtra("protocol") ?: "OPENAI"),
                            model = intent.getStringExtra("model") ?: "mock",
                            vision = intent.getBooleanExtra("vision", false),
                        )
                        g.providers.save(p, intent.getStringExtra("key"))
                        g.settings.setSelectedRemote(p.id)
                        "provider ${p.id} chosen (key kept: ${g.providers.hasKey(p.id)})"
                    }
                    "provider_off" -> {
                        (app as xyz.tifu.geode.GeodeApp).settings.setSelectedRemote("")
                        "the phone's model chosen"
                    }
                    "kvtest" -> kvTest(app as xyz.tifu.geode.GeodeApp) { s -> log.writeText("running kvtest\n$s") }
                    "speed" -> speedTest(app as xyz.tifu.geode.GeodeApp) { s -> log.writeText("running speed\n$s") }
                    "prompt" -> promptTest(app as xyz.tifu.geode.GeodeApp, intent.getIntExtra("words", 1500)) { s ->
                        log.writeText("running prompt\n$s")
                    }
                    else -> "unknown op"
                }
            }.getOrElse { it.stackTraceToString() }
            log.writeText("$text\n[${System.currentTimeMillis() - t0} ms]\n")
            Log.i("GeodeLinux", text.take(2000))
        }
    }

    /**
     * The conversation cache on the real model, greedy: a conversation's answer when it goes on from the storage (its
     * state saved after the first answer, the engine reset, the state loaded) against the same answer going on in RAM
     * (the cache off), and a new conversation's shared start (its long system prompt) loaded against read.
     */
    private suspend fun kvTest(app: xyz.tifu.geode.GeodeApp, progress: (String) -> Unit): String {
        val api = app.apiEngine
        val sb = StringBuilder()
        fun line(s: String) {
            sb.appendLine(s)
            progress(sb.toString())
        }
        fun msg(role: String, t: String, reasoning: String? = null) =
            xyz.tifu.geode.tokenizer.TemplateMessage(role, listOf(xyz.tifu.geode.tokenizer.Part.Text(t)), reasoning)
        val system = msg("system", (1..14).joinToString(" ") { i ->
            "Rule $i: answer briefly and precisely, in plain words, keeping every number exact, never inventing facts, " +
                "checking each step twice, and saying so when something is unknown or uncertain."
        })
        suspend fun ask(msgs: List<xyz.tifu.geode.tokenizer.TemplateMessage>, what: String): xyz.tifu.geode.server.GenResult {
            val t0 = System.currentTimeMillis()
            val r = api.run(xyz.tifu.geode.server.GenRequest(msgs, emptyList(), thinking = false, maxTokens = 40,
                temperature = 0f, topK = 1, seed = 7), "call_") {}
            line("$what: ${System.currentTimeMillis() - t0} ms, prompt ${r.promptTokens}, cached ${r.cachedTokens}: " +
                r.content.replace('\n', ' ').take(160))
            return r
        }
        suspend fun forget() = app.gate.use(Any()) { app.engine.reset() }
        val q1 = listOf(system, msg("user", "Count from 1 to 5, separated by commas."))
        val s0 = app.settings.flow.first().stateCacheMb
        // as an answer in the chat: the app kept alive (adb: cmd deviceidle tempwhitelist lets it start from here)
        xyz.tifu.geode.engine.AnswerService.start(app)
        try {
            app.settings.setStateCacheMb(4096)
            withContext(Dispatchers.IO) { app.stateCache.clear() }
            forget()
            // with the cache: the first answer read whole (its shared start saved), its conversation saved at once
            val a1 = ask(q1, "A1 cache (start read + saved)")
            app.gate.flush()
            forget()
            val q2 = q1 + msg("assistant", a1.content, a1.reasoning) + msg("user", "Now from 6 to 9.")
            val a2s = ask(q2, "A2 from storage")
            forget()
            val q3 = listOf(system, msg("user", "Name three colours."))
            val a3s = ask(q3, "A3 start from storage")
            line("kvcache: " + app.stateCache.dir.listFiles()?.joinToString { "${it.name} ${it.length() / 1024} KB" })
            // without it: the same answers going on in RAM, everything read
            app.settings.setStateCacheMb(0)
            forget()
            val a1r = ask(q1, "A1 RAM")
            val a2r = ask(q1 + msg("assistant", a1r.content, a1r.reasoning) + msg("user", "Now from 6 to 9."), "A2 RAM")
            forget()
            val a3r = ask(q3, "A3 read whole")
            line("same: A1 ${a1.content == a1r.content}, A2 ${a2s.content == a2r.content}, A3 ${a3s.content == a3r.content}")
        } finally {
            app.settings.setStateCacheMb(s0)
            xyz.tifu.geode.engine.AnswerService.stop(app)
        }
        return "ok\n$sb"
    }

    /**
     * The decode speed on this phone: two greedy answers of 64 tokens, each a new conversation - the first warms the
     * expert cache, the second is measured (decode tokens/s, what it wrote, and where its time went).
     */
    /** A long prompt through the app's own engine (its settings, its model): the prompt's speed and the profile, the
     *  NPU's line included - how a prompt reads in the real app process (adb: op prompt, words N). */
    private suspend fun promptTest(app: xyz.tifu.geode.GeodeApp, words: Int, progress: (String) -> Unit): String {
        val text = buildString {
            val w = "the river carried stones and light past the old mill while children counted boats ".split(' ')
            for (i in 0 until words) append(w[i % w.size]).append(if (i % 17 == 16) ". " else " ")
        }
        val m = xyz.tifu.geode.tokenizer.TemplateMessage("user",
            listOf(xyz.tifu.geode.tokenizer.Part.Text("Summarize in one sentence: $text")))
        xyz.tifu.geode.engine.AnswerService.start(app)
        try {
            val out = StringBuilder()
            for (round in 0 until 2) {   // the first opens the model and warms the cache
                app.gate.use(Any()) { app.engine.reset() }
                val t0 = System.currentTimeMillis()
                val r = app.apiEngine.run(xyz.tifu.geode.server.GenRequest(listOf(m), emptyList(), thinking = false,
                    maxTokens = 4, temperature = 0f, topK = 1, seed = 7), "call_") {}
                out.appendLine("round $round: ${r.promptTokens} prompt tokens at ${"%.1f".format(r.prefillTps)} tok/s, " +
                    "${System.currentTimeMillis() - t0} ms in all")
                progress(out.toString())
            }
            out.appendLine(app.gate.use(Any()) { runCatching { app.engine.profile() }.getOrElse { "$it" } })
            return "ok\n$out"
        } finally {
            xyz.tifu.geode.engine.AnswerService.stop(app)
        }
    }

    private suspend fun speedTest(app: xyz.tifu.geode.GeodeApp, progress: (String) -> Unit): String {
        val sb = StringBuilder()
        fun line(s: String) {
            sb.appendLine(s)
            progress(sb.toString())
        }
        fun msg(role: String, t: String) =
            xyz.tifu.geode.tokenizer.TemplateMessage(role, listOf(xyz.tifu.geode.tokenizer.Part.Text(t)))
        val prompts = listOf("Explain in a few sentences why the sky is blue.",
            "Поясни кількома реченнями, чому взимку холодніше, ніж улітку.")
        xyz.tifu.geode.engine.AnswerService.start(app)
        try {
            for ((i, q) in prompts.withIndex()) {
                app.gate.use(Any()) { app.engine.reset() }
                val t0 = System.currentTimeMillis()
                val r = app.apiEngine.run(xyz.tifu.geode.server.GenRequest(listOf(msg("user", q)), emptyList(),
                    thinking = false, maxTokens = 64, temperature = 0f, topK = 1, seed = 7), "call_") {}
                line("${if (i == 0) "warm" else "measured"}: decode ${"%.2f".format(r.decodeTps)} tok/s, " +
                    "${r.outputTokens} tokens in ${System.currentTimeMillis() - t0} ms: " +
                    r.content.replace('\n', ' ').take(120))
                // where the measured answer's time went (geode_profile: the phases, the speculation, the experts)
                if (i == 1) line(app.gate.use(Any()) { runCatching { app.engine.profile() }.getOrElse { "$it" } })
            }
        } finally {
            xyz.tifu.geode.engine.AnswerService.stop(app)
        }
        return "ok\n$sb"
    }

    companion object {
        private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    }
}
