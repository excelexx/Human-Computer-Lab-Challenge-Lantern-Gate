// CPU DOM scheduling fixtures, not browser/device or inference validation.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../src/checkin/live_camera.js'), 'utf8');
const bridgeKey = Symbol.for('checkin.liveCamera.bridge');

function fixture() {
  let nextId = 0;
  const intervals = new Map(), windowListeners = new Map(), observers = [];
  const events = [];
  class Input {
    constructor() { this.tagName = 'TEXTAREA'; this._value = ''; }
    get value() { return this._value; }
    set value(value) { this._value = value; }
    dispatchEvent(event) { events.push({type: event.type, value: this.value, bubbles: event.bubbles}); }
  }
  class Track {
    constructor() { this.readyState = 'live'; this.listeners = new Set(); this.stops = 0; }
    addEventListener(name, fn) { assert.equal(name, 'ended'); this.listeners.add(fn); }
    removeEventListener(name, fn) { this.listeners.delete(fn); }
    stop() { this.stops++; this.readyState = 'ended'; }
    endExternally() { this.readyState = 'ended'; [...this.listeners].forEach(fn => fn()); }
  }
  const makeStream = () => { const track = new Track(); return {track, getTracks: () => [track], getVideoTracks: () => [track]}; };
  const rootListeners = new Set();
  let recording = false, permissionVisible = true;
  const video = {srcObject: null, readyState: 2, videoWidth: 640, videoHeight: 480};
  const text = value => ({nodeType: 3, textContent: value});
  function button(label, permission = false) {
    const attributes = new Map([['aria-label', label]]);
    const container = {childNodes: [text(label)]};
    return {
      attributes, container, disabled: false, clicks: 0,
      getAttribute: name => attributes.get(name) || null,
      setAttribute: (name, value) => attributes.set(name, value),
      querySelector: () => container,
      closest(selector) { return selector === 'button' ? this : permission ? {} : null; },
      click() {
        this.clicks++;
        const event = {target: this, stopped: false, preventDefault() {}, stopImmediatePropagation() { this.stopped = true; }};
        [...rootListeners].forEach(fn => fn(event));
        if (event.stopped) return;
        if (permission) { video.srcObject = makeStream(); permissionVisible = false; recording = false; }
        else {
          recording = !recording;
          if (!recording) { video.srcObject?.getTracks().forEach(track => track.stop()); video.srcObject = null; permissionVisible = true; }
        }
      }
    };
  }
  const permission = button('Click to Access Webcam', true), capture = button('capture photo');
  permission.closest = selector => selector === 'button' ? permission : {};
  capture.closest = selector => selector === 'button' ? capture : null;
  const root = {
    querySelector(selector) {
      if (selector === 'video') return video;
      if (selector.includes('grant webcam')) return permissionVisible ? permission : null;
      if (selector.includes('capture photo')) return permissionVisible ? null : capture;
      return null;
    },
    contains: node => node === capture || node === permission,
    addEventListener: (name, fn) => rootListeners.add(fn),
    removeEventListener: (name, fn) => rootListeners.delete(fn)
  };
  const pill = {textContent: 'Joy', setAttribute(name, value) { this[name] = value; }}, note = {textContent: 'Old signal'};
  const panel = {querySelector: selector => selector === '.emotion-pill' ? pill : note, setAttribute() {}};
  const input = new Input();
  const env = {root, rootPresent: true, controlPresent: true};
  const document = {querySelector(selector) {
    if (selector === '#live-camera') return env.rootPresent ? env.root : null;
    if (selector.includes('#live-camera-control')) return env.controlPresent ? input : null;
    if (selector === '#live-camera-tag') return panel;
    return null;
  }};
  const window = {
    crypto: {randomUUID: () => `uuid-${++nextId}`}, HTMLTextAreaElement: Input, HTMLInputElement: Input,
    setInterval(fn) { const id = ++nextId; intervals.set(id, fn); return id; },
    clearInterval: id => intervals.delete(id),
    addEventListener(name, fn) { windowListeners.set(name, fn); },
    removeEventListener: name => windowListeners.delete(name)
  };
  class MutationObserver {
    constructor(fn) { this.fn = fn; this.disconnected = false; observers.push(this); }
    observe() {}
    disconnect() { this.disconnected = true; }
  }
  class Event { constructor(type, options) { this.type = type; Object.assign(this, options); } }
  const init = vm.runInNewContext(`(${source})`, {window, document, MutationObserver, Event, Symbol});
  return Object.assign(env, {init, window, input, events, video, capture, permission, pill, note, intervals, observers,
    tick: () => [...intervals.values()].forEach(fn => fn()),
    signal: name => windowListeners.get(name)?.(),
    listeners: rootListeners,
    makeStream,
    grant() { permission.click(); },
    started: () => recording
  });
}

test('permission remains a user gesture, then exactly one automatic start announces via native input', () => {
  const f = fixture(); f.init(); f.tick();
  assert.equal(f.permission.clicks, 0);
  assert.equal(f.permission.getAttribute('aria-label'), 'Turn camera on');
  assert.deepEqual(f.events, []);
  f.grant(); f.tick(); f.tick();
  assert.equal(f.capture.clicks, 1);
  assert.equal(f.started(), true);
  assert.equal(f.events.length, 1);
  assert.match(f.events[0].value, /^on:uuid-/);
  assert.equal(f.events[0].bubbles, true);
  assert.equal(f.capture.getAttribute('aria-label'), 'Turn camera off');
});

test('camera Stop stops tracks, announces off once and never auto-restarts without a new permission gesture', () => {
  const f = fixture(); f.init(); f.grant(); f.tick();
  const previous = f.video.srcObject;
  f.capture.click(); f.tick(); f.tick();
  assert.equal(previous.track.readyState, 'ended');
  assert.equal(f.started(), false);
  assert.deepEqual(f.events.map(e => e.value.split(':')[0]), ['on', 'off']);
  assert.equal(f.pill.textContent, 'Camera off');
  f.pill.textContent = 'Late old tag'; f.tick();
  assert.equal(f.pill.textContent, 'Camera off');
  f.grant(); f.tick();
  assert.equal(f.started(), true);
  assert.deepEqual(f.events.map(e => e.value.split(':')[0]), ['on', 'off', 'on']);
  assert.notEqual(f.events[0].value, f.events[2].value);
});

test('unready video, missing bridge input or disabled native button never start streaming', () => {
  const f = fixture(); f.init(); f.grant(); f.video.readyState = 1; f.tick();
  assert.equal(f.capture.clicks, 0);
  f.capture.click(); // Manual early press must not toggle native recording.
  assert.equal(f.started(), false);
  f.video.readyState = 2; f.controlPresent = false; f.tick();
  assert.equal(f.started(), false);
  f.controlPresent = true; f.capture.disabled = true; f.tick();
  assert.equal(f.started(), false);
  f.capture.disabled = false; f.tick();
  assert.equal(f.started(), true);
  assert.equal(f.events.length, 1);
});

test('external track ending stops native streaming and does not restart', () => {
  const f = fixture(); f.init(); f.grant(); f.tick();
  f.video.srcObject.track.endExternally(); f.tick();
  assert.equal(f.started(), false);
  assert.deepEqual(f.events.map(e => e.value.split(':')[0]), ['on', 'off']);
  assert.equal(f.pill.textContent, 'Camera off');
});

test('repeated initialization is idempotent and page exit removes observers/listeners/timers and stops tracks', () => {
  const f = fixture(); f.init(); f.init();
  assert.equal(f.intervals.size, 1);
  assert.equal(f.listeners.size, 1);
  assert.equal(f.observers.length, 1);
  f.grant(); f.tick(); const track = f.video.srcObject.track;
  f.signal('pagehide');
  assert.equal(track.readyState, 'ended');
  assert.equal(f.intervals.size, 0);
  assert.equal(f.listeners.size, 0);
  assert.equal(f.observers[0].disconnected, true);
  assert.equal(f.window[bridgeKey], undefined);
});

test('component removal disposes its camera and independent pages never share lifecycle flags', () => {
  const a = fixture(), b = fixture(); a.init(); b.init();
  a.grant(); a.tick(); const track = a.video.srcObject.track;
  a.rootPresent = false; a.tick();
  assert.equal(track.readyState, 'ended');
  assert.equal(a.listeners.size, 0);
  assert.deepEqual(b.events, []);
  b.grant(); b.tick();
  assert.equal(b.started(), true);
});
