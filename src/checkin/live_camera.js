() => {
  // Pinned to Gradio 5.49.1 Image/Webcam.svelte. Gradio still owns camera
  // permission, JPEG capture, and transport; this bridge never reads pixels.
  const key = Symbol.for("checkin.liveCamera.bridge");
  if (window[key]) {
    window[key].scan();
    return;
  }
  let controller = null;
  let disposed = false;
  const freshId = () => window.crypto.randomUUID();

  function controlInput() {
    return document.querySelector("#live-camera-control textarea, #live-camera-control input");
  }

  function announce(kind) {
    const input = controlInput();
    if (!input) return false;
    const prototype = input.tagName === "TEXTAREA" ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(prototype, "value").set;
    setter.call(input, `${kind}:${freshId()}`);
    input.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
    return true;
  }

  function clearSignal() {
    const panel = document.querySelector("#live-camera-tag");
    if (!panel) return;
    const pill = panel.querySelector(".emotion-pill");
    const note = panel.querySelector(".emotion-note");
    if (pill) {
      pill.textContent = "Camera off";
      pill.setAttribute("title", "Turn on your camera for an emotion tag.");
      pill.setAttribute("data-held", "false");
    }
    if (note) note.textContent = "Turn the camera on for a live emotion signal.";
    panel.setAttribute("data-camera-active", "false");
  }

  function replaceLabel(button, label) {
    if (button.getAttribute("aria-label") !== label) button.setAttribute("aria-label", label);
    if (button.getAttribute("title") !== label) button.setAttribute("title", label);
    const container = button.querySelector(".icon-with-text, .wrap") || button;
    for (const node of container.childNodes) {
      if (node.nodeType === 3 && node.textContent.trim() && node.textContent.trim() !== label) node.textContent = ` ${label} `;
    }
  }

  function attach(root) {
    let wanted = true;
    let activeStream = null;
    let started = false;
    let programmatic = false;
    let busy = false;
    let closed = false;
    const endedListeners = new Map();
    const blockedStreams = new WeakSet();
    const toggle = () => root.querySelector('button[aria-label="capture photo"], button[data-checkin-capture]');

    function forgetTracks() {
      for (const [track, listener] of endedListeners) track.removeEventListener("ended", listener);
      endedListeners.clear();
    }

    function off({ nativeClick = false, keepWanted = false } = {}) {
      const previous = activeStream;
      const wasActive = started || previous !== null;
      if (!keepWanted) wanted = false;
      if (previous) blockedStreams.add(previous);
      forgetTracks();
      if (nativeClick && started) {
        const button = toggle();
        if (button) {
          programmatic = true;
          try { button.click(); } finally { programmatic = false; }
        }
      }
      // Calling stop is idempotent; the native Stop also stops these tracks.
      for (const track of previous?.getTracks() || []) track.stop();
      activeStream = null;
      started = false;
      if (wasActive) announce("off");
      clearSignal();
    }

    function scan() {
      if (closed || busy) return;
      busy = true;
      try {
        const permission = root.querySelector('[title="grant webcam access"] button');
        if (permission) replaceLabel(permission, "Turn camera on");
        const button = toggle();
        if (button) {
          button.setAttribute("data-checkin-capture", "true");
          replaceLabel(button, started ? "Turn camera off" : "Start tracking");
        }
        const video = root.querySelector("video");
        const stream = video?.srcObject;
        const tracks = stream?.getVideoTracks?.() || [];
        const live = tracks.some(track => track.readyState === "live");
        if (activeStream && (stream !== activeStream || !live)) off({ nativeClick: stream === activeStream, keepWanted: stream !== activeStream });
        if (!wanted) { clearSignal(); return; }
        if (!live || !button || button.disabled || !controlInput() || blockedStreams.has(stream)) return;
        if (video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0 || started) return;
        activeStream = stream;
        for (const track of tracks) {
          const listener = () => { if (activeStream === stream) off({ nativeClick: true }); };
          endedListeners.set(track, listener);
          track.addEventListener("ended", listener);
        }
        if (!announce("on")) return;
        started = true; // Set before click to survive synchronous mutations.
        programmatic = true;
        try { button.click(); } finally { programmatic = false; }
        replaceLabel(button, "Turn camera off");
        const panel = document.querySelector("#live-camera-tag");
        if (panel) panel.setAttribute("data-camera-active", "true");
      } finally {
        busy = false;
      }
    }

    function clicked(event) {
      const button = event.target.closest?.("button");
      if (!button || !root.contains(button) || programmatic) return;
      if (button.closest('[title="grant webcam access"]')) {
        wanted = true; // Only a new explicit permission gesture re-arms Stop.
        return;
      }
      if (button === toggle()) {
        if (started) off();
        else {
          // A quick manual press during camera startup must not toggle the
          // native recording flag once here and again in the automatic scan.
          event.preventDefault();
          event.stopImmediatePropagation();
          scan();
        }
      }
    }

    const observer = new MutationObserver(scan);
    observer.observe(root, { childList: true, subtree: true, attributes: true, attributeFilter: ["class", "disabled", "src"] });
    root.addEventListener("click", clicked, true);
    scan();
    return {
      root,
      scan,
      dispose() {
        if (closed) return;
        closed = true;
        observer.disconnect();
        root.removeEventListener("click", clicked, true);
        off({ nativeClick: true });
      }
    };
  }

  function scan() {
    if (disposed) return;
    const root = document.querySelector("#live-camera");
    if (root !== controller?.root) {
      controller?.dispose();
      controller = root ? attach(root) : null;
    }
    controller?.scan();
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    window.clearInterval(timer);
    window.removeEventListener("pagehide", dispose);
    window.removeEventListener("beforeunload", dispose);
    controller?.dispose();
    controller = null;
    delete window[key];
  }

  // Property changes such as video.srcObject do not emit DOM mutations.
  // A modest poll complements the scoped observer and survives component swaps.
  const timer = window.setInterval(scan, 200);
  window[key] = { scan, dispose };
  window.addEventListener("pagehide", dispose);
  window.addEventListener("beforeunload", dispose);
  scan();
}
