// ===========================
// AgrowAssist — chatbot.js
// ===========================

document.addEventListener("DOMContentLoaded", function () {
  initChatbot();
});

function initChatbot() {
  const toggle = document.getElementById("chatbot-toggle");
  const panel = document.getElementById("chatbot-panel");
  const closeBtn = document.getElementById("chatbot-close");
  const clearBtn = document.getElementById("chatbot-clear");
  const messagesEl = document.getElementById("chatbot-messages");
  const form = document.getElementById("chatbot-form");
  const input = document.getElementById("chatbot-input");
  const sendBtn = document.getElementById("chatbot-send");
  const quickPrompts = document.getElementById("chatbot-quick-prompts");

  if (!toggle || !panel) return;

  const STORAGE_KEY = "agrowassist_chat_history";
  const MAX_STORED_MESSAGES = 40;

  let history = loadHistory();
  let isWaiting = false;

  renderAllMessages();

  toggle.addEventListener("click", () => {
    panel.classList.toggle("is-open");
    if (panel.classList.contains("is-open")) {
      input.focus();
      scrollToBottom();
    }
  });

  closeBtn.addEventListener("click", () => {
    panel.classList.remove("is-open");
  });

  clearBtn.addEventListener("click", () => {
    history = [];
    localStorage.removeItem(STORAGE_KEY);
    messagesEl.innerHTML = "";
    renderAllMessages();
  });

  quickPrompts.addEventListener("click", (e) => {
    const chip = e.target.closest(".chatbot-chip");
    if (!chip) return;
    input.value = chip.dataset.prompt;
    form.requestSubmit();
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message || isWaiting) return;

    addMessage("user", message);
    input.value = "";
    setWaiting(true);
    const typingEl = addTypingIndicator();

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: message,
          history: history.slice(0, -1), // history before this new user message
        }),
      });

      const data = await response.json();
      typingEl.remove();

      if (!response.ok) {
        addMessage(
          "bot",
          data.error || "Something went wrong. Please try again.",
          true,
        );
      } else {
        addMessage("bot", data.reply);
      }
    } catch (err) {
      typingEl.remove();
      addMessage(
        "bot",
        "Couldn't reach the server. Check your connection and try again.",
        true,
      );
    } finally {
      setWaiting(false);
    }
  });

  function setWaiting(waiting) {
    isWaiting = waiting;
    sendBtn.disabled = waiting;
    input.disabled = waiting;
  }

  function addMessage(role, text, isError = false) {
    history.push({ role: role, text: text });
    if (history.length > MAX_STORED_MESSAGES) {
      history = history.slice(-MAX_STORED_MESSAGES);
    }
    saveHistory();
    renderMessage(role, text, isError);
    toggleQuickPrompts();
    scrollToBottom();
  }

  function renderMessage(role, text, isError = false) {
    const bubble = document.createElement("div");
    bubble.className = `chatbot-msg from-${role}${isError ? " is-error" : ""}`;
    bubble.textContent = text;
    messagesEl.appendChild(bubble);
  }

  function addTypingIndicator() {
    const el = document.createElement("div");
    el.className = "chatbot-msg from-bot chatbot-typing";
    el.innerHTML = "<span></span><span></span><span></span>";
    messagesEl.appendChild(el);
    scrollToBottom();
    return el;
  }

  function renderAllMessages() {
    history.forEach((m) => renderMessage(m.role, m.text));
    toggleQuickPrompts();
    scrollToBottom();
  }

  function toggleQuickPrompts() {
    quickPrompts.style.display = history.length === 0 ? "flex" : "none";
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    });
  }

  function loadHistory() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (e) {
      return [];
    }
  }

  function saveHistory() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (e) {
      // storage full or unavailable — chat still works, just won't persist
    }
  }
}
