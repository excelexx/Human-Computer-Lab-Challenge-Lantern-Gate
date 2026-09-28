"""Offline pixel-world shell around the existing local multimodal controls."""
import base64
from pathlib import Path

MARA_PORTRAIT = '''<div class="mara-portrait"><svg viewBox="0 0 64 76" role="img" aria-label="Mara, the gatekeeper, holding her lantern" shape-rendering="crispEdges" xmlns="http://www.w3.org/2000/svg">

<path d="M18 17h4V9h24v7h5v29h-8V29H21v16h-7V23h4z" fill="#5b403b"/>
<path d="M23 16h20v4h5v19h-5v6H24v-5h-5V23h4z" fill="#e8b78d"/>
<path d="M21 15h6v-3h17v6h-7v4h-9v4h-9v-8h2z" fill="#ba7e50"/>
<path d="M23 27h6v2h-6zm14 0h6v2h-6z" fill="#755345"/>
<path d="M25 30h3v4h-3zm13 0h3v4h-3z" fill="#303c40"/>
<path d="M30 39h9v2h-9z" fill="#a66958"/>
<path d="M20 46h7v-4h14v4h6v5h7v25H10V55h10z" fill="#4e786d"/>
<path d="M27 45h14v6H27zM18 52h4v24h-4zm27 0h4v24h-4z" fill="#35564e"/>
<path d="M24 46h6v6h6v-6h7v7h-9v5h-5v-5h-5z" fill="#ebcd89"/>
<path d="M46 59h10v7H46zM12 63h7v9h-7z" fill="#e8b78d"/>
<path d="M52 56h9v3h-9zm-2 3h13v16H50zm3-3v-5h7v5h-2v-3h-3v3z" fill="#775744"/>
<path d="M53 62h7v10h-7z" fill="#ffe3a0"/><path d="M55 64h3v6h-3z" fill="#fff2c6"/>
</svg><span>Mara</span></div>'''


def asset_uri(name, kind):
    payload = (Path(__file__).with_name('assets') / name).read_bytes()
    return f'data:{kind};base64,' + base64.b64encode(payload).decode('ascii')


def game_html():
    atlas = asset_uri('tilemap_packed.png', 'image/png')
    return f'''<div id="game-world">
      <canvas id="town-canvas" tabindex="0" aria-label="Lantern Gate village. Move with W A S D or arrow keys. Approach Mara at the northern gate to talk." data-atlas="{atlas}"></canvas>
      <div class="world-header"><div class="quest-plaque"><h1>Lantern Gate</h1><p id="game-hint" role="status">Find Mara at the northern gate. Walk up the stone path.</p></div>
      <button id="game-fullscreen" title="Toggle browser full screen" aria-label="Toggle full screen">Full screen</button></div>
      <div class="world-footer"><div class="world-controls"><span><kbd>E</kbd> Talk</span><span><kbd>Esc</kbd> Leave</span></div>
      <button id="visit-mara">Go to Mara</button><button id="talk-mara" hidden>Talk to Mara</button><button id="restart-scene" hidden>Restart scene</button></div>
      <div class="touch-pad" aria-label="Movement controls"><button data-move="w" aria-label="Walk north">▲</button><div><button data-move="a" aria-label="Walk west">◀</button><button data-move="s" aria-label="Walk south">▼</button><button data-move="d" aria-label="Walk east">▶</button></div></div>
      <div class="asset-credit">Town art: Kenney · Local AI · No audio</div>
    </div>'''


def game_css():
    font = asset_uri('Silkscreen-Regular.ttf', 'font/ttf')
    return f"@font-face {{font-family:PixelTown;src:url('{font}') format('truetype');font-display:swap;}}\n" + GAME_CSS + COMPACT_CSS + SKETCH_CSS + FINAL_LAYOUT_CSS + IN_WORLD_CSS + FLOATING_CSS


GAME_CSS = '''
:root,body,.dark {background:#253c46 !important;}
body.game-ready {overflow:hidden !important;}
body.game-ready .gradio-container.gradio-container {padding:0 !important;max-width:none !important;min-height:100dvh;background:transparent !important;}
#game-world {position:fixed;inset:0;z-index:1;overflow:hidden;background:#253c46;font-family:PixelTown,Consolas,monospace;color:#fff0c2;}
#town-canvas {display:block;width:100%;height:100%;image-rendering:pixelated;image-rendering:crisp-edges;outline:none;}
#town-canvas:focus-visible {outline:3px solid #fff0c2;outline-offset:-4px;}
.world-header,.world-footer {position:absolute;left:24px;right:24px;display:flex;gap:12px;align-items:flex-start;justify-content:space-between;pointer-events:none;}
.world-header {top:22px;}.world-footer {bottom:24px;align-items:center;justify-content:flex-start;flex-wrap:wrap;}
.quest-plaque {max-width:440px;background:#28343aee;border:3px solid #d0b17d;box-shadow:4px 4px 0 #26343b;padding:16px 20px;}
.quest-plaque h1 {font:20px PixelTown,monospace;margin:0 0 10px;color:#fff0c2;}
.quest-plaque p {font:12px/1.7 PixelTown,monospace;margin:0;color:#ead9b0;}
#game-world button {pointer-events:auto;font:11px/1.5 PixelTown,monospace;background:#fff0c2;color:#403834;border:3px solid #685246;box-shadow:3px 3px 0 #26343b;border-radius:0;padding:10px 13px;cursor:pointer;}
#game-world button:hover {background:#f4d598;}#game-world button:focus-visible {outline:3px solid #fff;outline-offset:4px;}
.world-controls {display:flex;gap:15px;flex-wrap:wrap;padding:12px;background:#28343aee;border:2px solid #b99b72;font-size:10px;}
.world-controls kbd {font:11px PixelTown,monospace;background:#fff0c2;color:#403834;padding:3px;margin-right:3px;}
.touch-pad {position:absolute;right:22px;bottom:72px;display:flex;flex-direction:column;align-items:center;gap:4px;}
.touch-pad div {display:flex;gap:4px;}.touch-pad button {touch-action:none;min-width:40px;min-height:40px;padding:6px !important;}
.asset-credit {position:absolute;bottom:5px;right:12px;font:9px PixelTown,monospace;color:#f5e4ba;text-shadow:1px 1px #24313a;}
#dialogue-panel[data-game-open="false"] {display:none !important;}
#dialogue-panel[data-game-open="true"] {display:flex !important;}
#game-world[data-dialogue="true"]::after {content:'';position:absolute;inset:0;background:#1425306b;pointer-events:none;}
#game-world[data-dialogue="true"] .world-header,#game-world[data-dialogue="true"] .world-footer,#game-world[data-dialogue="true"] .touch-pad {visibility:hidden;}
#dialogue-panel {position:fixed;z-index:10;left:50%;top:50%;transform:translate(-50%,-50%);width:min(1050px,calc(100vw - 40px));max-height:calc(100dvh - 40px);overflow:auto !important;background:#fff0d1 !important;border:4px solid #5d4940 !important;box-shadow:0 0 0 3px #d8b67c,9px 9px 0 #1c2c32;border-radius:0 !important;padding:18px !important;gap:10px !important;color:#453c36 !important;}
#dialogue-panel,#dialogue-panel * {font-family:PixelTown,Consolas,monospace !important;}
.dialogue-title {display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid #c5ac88;padding:0 4px 12px;color:#4c443c;font-size:16px;gap:10px;}
#close-dialogue {font:11px PixelTown,monospace;color:#4c443c;background:#ead1a2;border:2px solid #826954;padding:9px 12px;cursor:pointer;}
#dialogue-panel #content-grid {gap:18px !important;align-items:flex-start !important;}
#dialogue-panel #input-column,#dialogue-panel #conversation-column {background:transparent !important;border:0;box-shadow:none;padding:6px;gap:10px;}
#dialogue-panel .panel-heading h2 {font-size:14px;color:#4c443c;margin-bottom:9px;}
#dialogue-panel .panel-heading p {font-size:10px;color:#73624f;line-height:1.7;}
#dialogue-panel #input-column .tab-nav button {font-size:9px;color:#5d4a38 !important;}
#dialogue-panel #input-column .tab-nav button.selected {color:#304d4e !important;border-bottom-color:#304d4e !important;}
#dialogue-panel #live-camera,#dialogue-panel #camera-clip {background:#3b4b4d !important;border:3px solid #857354 !important;height:180px !important;min-height:180px !important;}
#dialogue-panel #live-camera video {background:#2a3c42;}
#dialogue-panel #live-camera .button-wrap {border-radius:0;color:#fff0c2;}
#dialogue-panel #live-camera .wrap {color:#fff0c2;}
#dialogue-panel #checkin-message textarea {background:#fff8e7 !important;color:#493d33 !important;border:2px solid #a28a68 !important;font-size:12px !important;line-height:1.7;padding:10px !important;}
#dialogue-panel #checkin-message textarea::placeholder {color:#826e58 !important;}
#dialogue-panel label,#dialogue-panel .block-title,#dialogue-panel .block-label {background:transparent !important;color:#514537 !important;font-size:11px !important;}
#dialogue-panel #conversation {background:#e8d8b6 !important;border:2px solid #b69a71 !important;height:210px !important;}
#dialogue-panel #conversation .user {background:#d3dabc !important;color:#394536 !important;}
#dialogue-panel #conversation .bot {background:#f6e8cb !important;color:#493f34 !important;}
#dialogue-panel #conversation .message {color:#493f34 !important;font-size:12px;line-height:1.8;}
#dialogue-panel .conversation-empty h3 {color:#5b4937;font-size:15px;}
#dialogue-panel .conversation-empty p {color:#715e4a;font-size:11px;line-height:1.9;}
#dialogue-panel .conversation-empty .speaker {display:none;}
#dialogue-panel .sample-line {background:#eee0bf !important;color:#5b4a37 !important;border:2px solid #bba17a !important;font-size:10px !important;min-height:34px !important;line-height:1.6;}
#dialogue-panel .sample-line:hover {background:#f8edcc !important;border-color:#7c684f !important;}
#dialogue-panel #sample-note p,#dialogue-panel #capture-note p {font-size:9px !important;line-height:1.8 !important;color:#72604b !important;}
#dialogue-panel .emotion-pill {font-size:10px;color:#3c574c;background:#d8dfc2;border:2px solid #87967b;padding:4px 8px;}
#dialogue-panel .emotion-caption {font-size:9px;color:#76624b;}
#dialogue-panel .emotion-note {font-size:9px;line-height:1.8;color:#72604b;}
#dialogue-panel #live-camera-tag {min-height:46px;}
#dialogue-panel #send {font-size:12px;min-height:43px;color:#fff0c2 !important;background:#4e6b5b !important;border:2px solid #304e44 !important;box-shadow:3px 3px 0 #a48a67 !important;}
#dialogue-panel #stop {font-size:10px;min-height:43px;color:#6a5543 !important;background:#e8d6b5 !important;border:2px solid #b29b7c !important;}
#dialogue-panel #emotion-panel {border-color:#c8b18c;padding:10px 0 0;}
#dialogue-panel #activity p,#dialogue-panel #care-note p {font-size:9px !important;line-height:1.8 !important;color:#72604b !important;}
#dialogue-panel #new-conversation {font-size:10px !important;color:#5e6d4d !important;}
#dialogue-panel #diagnostics {background:#2a3f4b !important;color:#dfe9ef !important;margin-top:0;border-radius:0;}
#dialogue-panel #diagnostics * {font-family:Consolas,monospace !important;}
#dialogue-panel button:focus-visible,#dialogue-panel textarea:focus-visible {outline:3px solid #456e75 !important;}
@media(min-width:900px){.touch-pad{opacity:.85;}}
@media(max-width:800px){
 .world-header{left:12px;right:12px;top:12px;gap:8px;}.quest-plaque{padding:10px 12px;max-width:72%;}.quest-plaque h1{font-size:14px;}.quest-plaque p{font-size:9px;}
 #game-fullscreen{font-size:9px !important;padding:7px !important;}.world-footer{left:12px;right:12px;bottom:25px;max-width:65%;}.world-controls{font-size:8px;gap:10px;line-height:1.8;}.world-controls kbd{font-size:9px;}.touch-pad{right:12px;bottom:50px;}
 #dialogue-panel{width:calc(100vw - 20px);max-height:calc(100dvh - 20px);padding:10px !important;}.dialogue-title{font-size:12px;}#close-dialogue{font-size:9px;}
 #dialogue-panel #input-column,#dialogue-panel #conversation-column{padding:3px !important;}#dialogue-panel #content-grid{gap:12px !important;}
}
'''

COMPACT_CSS = '''
.dialogue-title span {color:#4c443c !important;}
#dialogue-panel #conversation .bubble-wrap {background:transparent !important;}
#dialogue-panel #input-column [role="tablist"],#dialogue-panel #input-column .tab-container {display:none !important;}
#dialogue-panel #checkin-message label.container {background:transparent !important;border:0 !important;border-radius:0 !important;}
#dialogue-panel #checkin-message label.container > span {background:transparent !important;color:#69553d !important;}
#dialogue-panel #checkin-message .input-container {background:transparent !important;border-radius:0 !important;}
#dialogue-panel #live-camera button > .wrap {height:100% !important;min-height:0 !important;gap:8px !important;padding:8px !important;}
#dialogue-panel #live-camera button > .wrap svg {max-height:26px;max-width:26px;}
#dialogue-panel #live-camera button {color:#fff0c2 !important;font-size:10px !important;}
#dialogue-panel #live-camera button span {color:#fff0c2 !important;}
#dialogue-panel {width:min(860px,calc(100vw - 36px));padding:16px !important;gap:12px !important;}
.dialogue-title {font-size:13px;padding-bottom:10px;}
#dialogue-panel #content-grid {gap:15px !important;}
#dialogue-panel #input-column {flex:0 0 205px !important;min-width:0 !important;max-width:205px !important;}
#dialogue-panel #input-column .tab-nav {display:none !important;}
#dialogue-panel #input-column .tabs {border:0 !important;}
#dialogue-panel #input-column .tabitem {padding:0 !important;}
#dialogue-panel #live-camera {height:153px !important;min-height:153px !important;}
#dialogue-panel #live-camera-tag .emotion-note {display:none;}
#dialogue-panel #live-camera-tag {min-height:28px;}
#dialogue-panel #live-camera-tag .emotion-line {gap:5px;}
#dialogue-panel #live-camera-tag .emotion-caption {font-size:9px;}
#dialogue-panel #conversation-column {display:grid !important;grid-template-columns:155px minmax(0,1fr);grid-template-rows:180px auto;gap:8px !important;flex:1 1 0% !important;min-width:0 !important;}
#mara-portrait {grid-column:1;grid-row:1 / span 2;padding:0 !important;}
.mara-portrait {width:100%;height:100%;display:flex;flex-direction:column;align-items:center;}
.mara-portrait svg {width:150px;height:178px;max-width:100%;image-rendering:pixelated;border:3px solid #8c7859;box-shadow:3px 3px 0 #ceb78a;}
.mara-portrait span {font:12px PixelTown,monospace;color:#594835;padding-top:8px;}
#dialogue-panel #conversation {height:180px !important;min-height:180px !important;grid-column:2;grid-row:1;border:0 !important;background:transparent !important;}
#dialogue-panel #conversation .bot {background:transparent !important;}
#dialogue-panel #conversation .user {display:none !important;}
#dialogue-panel #conversation button {display:none !important;}
#dialogue-panel #conversation .message {font-size:11px;line-height:1.9;}
#dialogue-panel .conversation-empty p {font-size:11px;line-height:1.9;color:#554633;}
#dialogue-panel #activity {grid-column:2;grid-row:2;min-height:0;}
#dialogue-panel #activity p {font-size:8px !important;line-height:1.6 !important;margin:0;}
#dialogue-panel #player-replies {border-top:2px solid #c6b089;padding-top:12px;gap:8px !important;}
#dialogue-panel #player-replies .sample-row {gap:8px !important;}
#dialogue-panel .sample-line {min-height:36px !important;text-align:left !important;font-size:11px !important;padding:7px 12px !important;}
#dialogue-panel #custom-reply-row {gap:8px !important;align-items:flex-end !important;}
#dialogue-panel #checkin-message textarea {min-height:42px !important;max-height:74px;font-size:11px !important;padding:10px !important;}
#dialogue-panel #send,#dialogue-panel #stop {min-height:42px !important;height:42px !important;flex-grow:0 !important;}
@media(max-width:800px){
 #dialogue-panel #content-grid{flex-direction:row !important;align-items:flex-start !important;}
 #dialogue-panel #input-column{flex:0 0 145px !important;max-width:145px !important;}
 #dialogue-panel #live-camera{height:125px !important;min-height:125px !important;}
 #dialogue-panel #conversation-column{flex:1 1 0% !important;grid-template-columns:115px minmax(0,1fr);grid-template-rows:160px auto;}
 .mara-portrait svg{width:110px;height:131px;}
 #dialogue-panel #conversation{height:160px !important;min-height:160px !important;}
}
@media(max-width:600px){
 #dialogue-panel{padding:10px !important;gap:8px !important;}
 .dialogue-title{font-size:10px;}.dialogue-title span{max-width:19ch;}#close-dialogue{font-size:8px;padding:6px;}
 #dialogue-panel #content-grid{gap:8px !important;}
 #dialogue-panel #input-column{flex:0 0 105px !important;max-width:105px !important;}
 #dialogue-panel #live-camera{height:105px !important;min-height:105px !important;}
 #dialogue-panel #live-camera label{display:none !important;}
 #dialogue-panel #live-camera .button-wrap{padding:6px !important;font-size:9px;}
 #dialogue-panel #conversation-column{display:flex !important;flex-direction:column !important;gap:4px !important;}
 .mara-portrait svg{width:95px;height:113px;}.mara-portrait span{font-size:10px;padding-top:4px;}
 #dialogue-panel #conversation{height:125px !important;min-height:125px !important;}
 #dialogue-panel #conversation .message,#dialogue-panel .conversation-empty p{font-size:9px;line-height:1.8;}
 #dialogue-panel .sample-line{font-size:9px !important;min-height:34px !important;padding:6px !important;}
 #dialogue-panel #send,#dialogue-panel #stop{min-width:60px !important;font-size:9px !important;}
}
'''

SKETCH_CSS = '''
#leave-dialogue {display:none !important;}
.touch-pad button {font-size:0 !important;display:grid;place-items:center;}
.touch-pad button::before {content:'';display:block;width:4px;height:4px;background:#50483e;box-shadow:0 -8px #50483e,0 -4px #50483e,-4px -4px #50483e,-8px 0 #50483e,4px -4px #50483e,8px 0 #50483e,0 4px #50483e,0 8px #50483e;}
.touch-pad button[data-move="a"]::before {transform:rotate(-90deg);}.touch-pad button[data-move="s"]::before {transform:rotate(180deg);}.touch-pad button[data-move="d"]::before {transform:rotate(90deg);}
#dialogue-panel[data-game-open="true"] {display:grid !important;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);grid-template-rows:auto auto 1fr;gap:14px 24px !important;width:min(900px,calc(100vw - 36px));}
#dialogue-header {grid-column:1 / -1;grid-row:1;padding:0 !important;}
#dialogue-panel #content-grid {display:contents !important;}
#dialogue-panel #input-column {grid-column:1;grid-row:2;max-width:none !important;width:100% !important;padding:0 !important;}
#dialogue-panel #input-column .tabitem {padding:0 !important;}
#dialogue-panel #live-camera {height:216px !important;min-height:216px !important;}
#dialogue-panel #live-camera .image-container {height:100%;}
#dialogue-panel #live-camera-tag {padding:8px 0 0;}
#dialogue-panel #player-replies {grid-column:1;grid-row:3;align-self:start;padding-top:10px;}
#dialogue-panel #conversation-column {grid-column:2;grid-row:2 / span 2;display:flex !important;flex-direction:column !important;align-items:stretch !important;padding:0 !important;width:100% !important;gap:10px !important;}
#dialogue-panel #mara-portrait {flex:0 0 auto;}
.mara-portrait svg {width:192px;height:228px;max-width:100%;}
.mara-portrait span {font-size:13px;}
#dialogue-panel #conversation {height:155px !important;min-height:155px !important;border:2px solid #c0a67a !important;background:#f7e8c6 !important;}
#dialogue-panel #conversation .message,#dialogue-panel .conversation-empty p {font-size:11px !important;line-height:1.9;}
#dialogue-panel #activity {min-height:12px;margin:0;}
#dialogue-panel #stop {display:none !important;}
#dialogue-panel #custom-reply-row {flex-wrap:nowrap !important;align-items:flex-end !important;}
#dialogue-panel #send {font-size:0 !important;min-width:43px !important;max-width:43px !important;width:43px !important;padding:0 !important;display:grid;place-items:center;}
#send::before {content:'';display:block;width:4px;height:4px;background:#fff0c2;box-shadow:-8px 0 #fff0c2,-4px 0 #fff0c2,4px 0 #fff0c2,8px 0 #fff0c2,4px -4px #fff0c2,0 -8px #fff0c2,4px 4px #fff0c2,0 8px #fff0c2;}
#dialogue-panel #checkin-message {min-width:0 !important;}
#dialogue-panel .sample-line {font-size:10px !important;min-height:40px !important;}
@media(max-width:600px){
 #dialogue-panel[data-game-open="true"]{width:calc(100vw - 20px);grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:8px !important;}
 #dialogue-panel #live-camera{height:135px !important;min-height:135px !important;}
 #dialogue-panel #input-column{max-width:none !important;}
 .mara-portrait svg{width:128px;height:152px;}
 #dialogue-panel #conversation{height:175px !important;min-height:175px !important;}
 #dialogue-panel #conversation .message,#dialogue-panel .conversation-empty p{font-size:9px !important;line-height:1.9;}
 #dialogue-panel #player-replies .sample-row{flex-direction:column !important;gap:5px !important;}
 #dialogue-panel .sample-line{font-size:9px !important;min-height:32px !important;}
 #dialogue-panel #send{min-width:34px !important;max-width:34px !important;}
 #dialogue-panel #checkin-message textarea{font-size:9px !important;padding:7px !important;}
 .dialogue-title span{max-width:18ch;}
}
'''

FINAL_LAYOUT_CSS = '''
#dialogue-panel #custom-reply-row .form {background:transparent !important;border:0 !important;border-radius:0 !important;box-shadow:none !important;padding:0 !important;}
#dialogue-panel #conversation .bot-row:not(:last-child) {display:none !important;}
#dialogue-panel[data-game-open="true"] {grid-template-columns:minmax(0,.9fr) minmax(0,1.2fr);width:min(960px,calc(100vw - 36px));gap:16px 28px !important;}
#dialogue-panel #input-column {grid-column:2;grid-row:2;}
#dialogue-panel #player-replies {grid-column:2;grid-row:3;}
#dialogue-panel #conversation-column {grid-column:1;grid-row:2 / span 2;}
#dialogue-panel #live-camera {height:270px !important;min-height:270px !important;}
#dialogue-panel #live-camera video {width:100% !important;height:100% !important;object-fit:cover !important;}
#dialogue-panel #conversation {order:0;height:205px !important;min-height:205px !important;position:relative;overflow:visible !important;background:#fff9e7 !important;border:3px solid #a18a64 !important;border-radius:3px;box-shadow:4px 4px 0 #dfc697;}
#dialogue-panel #conversation::after {content:'';position:absolute;bottom:-13px;left:35px;width:20px;height:14px;background:#a18a64;clip-path:polygon(0 0,100% 0,100% 30%,70% 30%,70% 60%,40% 60%,40% 100%,0 100%);}
#dialogue-panel #conversation .bubble-wrap {padding:8px !important;background:transparent !important;}
#dialogue-panel #conversation .message-wrap {width:100% !important;margin:0 !important;padding:0 !important;}
#dialogue-panel #conversation .message-row {width:100% !important;max-width:100% !important;margin:0 !important;}
#dialogue-panel #conversation .user-row {display:none !important;}
#dialogue-panel #conversation .flex-wrap {width:100% !important;max-width:100% !important;}
#dialogue-panel #conversation .bot {width:100% !important;max-width:100% !important;padding:8px !important;box-sizing:border-box !important;}
#dialogue-panel #conversation .message-content {width:100% !important;overflow-wrap:break-word;}
#dialogue-panel #conversation .message,#dialogue-panel #conversation .message * {color:#493c2c !important;font-size:12px !important;line-height:1.9 !important;}
#dialogue-panel #activity {order:1;padding-top:6px;}
#dialogue-panel #mara-portrait {order:2;margin-top:auto;align-self:flex-start;}
.mara-portrait {align-items:flex-start;}.mara-portrait svg {width:192px;height:228px;border:0;box-shadow:none;background:transparent;}.mara-portrait span {padding-left:64px;}
#dialogue-panel #checkin-message {padding:0 !important;border:0 !important;overflow:visible !important;background:transparent !important;box-shadow:none !important;}
#dialogue-panel #checkin-message label.container {display:flex;flex-direction:column;gap:5px;padding:0 !important;margin:0 !important;box-shadow:none !important;}
#dialogue-panel #checkin-message label.container > span {position:static !important;display:block !important;padding:0 !important;font-size:10px !important;line-height:1.4;color:#6b543c !important;}
#dialogue-panel #checkin-message textarea {border:2px solid #ad9267 !important;box-sizing:border-box !important;min-height:45px !important;height:45px;background:#fff9e7 !important;color:#493c2c !important;font-size:11px !important;line-height:1.6 !important;border-radius:0 !important;resize:none !important;overflow-y:auto !important;}
#dialogue-panel #checkin-message textarea::placeholder {color:#826b50 !important;opacity:1;}
#dialogue-panel #send {width:45px !important;min-width:45px !important;max-width:45px !important;min-height:45px !important;height:45px !important;margin:0 !important;background:#536a53 !important;color:#fff8df !important;border:2px solid #354f3f !important;border-radius:0 !important;box-shadow:2px 2px 0 #c3a779 !important;align-self:flex-end !important;}
#dialogue-panel #send:hover {background:#627d5f !important;}#dialogue-panel #send:disabled {opacity:.5;}
#dialogue-panel #send::before {margin-left:-3px;}
@media(max-width:600px){
 #dialogue-panel[data-game-open="true"] {grid-template-columns:minmax(0,.9fr) minmax(0,1.1fr);gap:10px !important;}
 #dialogue-panel #live-camera {height:168px !important;min-height:168px !important;}
 #dialogue-panel #conversation {height:225px !important;min-height:225px !important;}
 #dialogue-panel #conversation .message,#dialogue-panel #conversation .message * {font-size:9px !important;line-height:1.8 !important;}
 #dialogue-panel #conversation .bubble-wrap,#dialogue-panel #conversation .bot {padding:4px !important;}
 .mara-portrait svg {width:128px;height:152px;}.mara-portrait span{padding-left:32px;}
 #dialogue-panel #checkin-message textarea {font-size:9px !important;}
 #dialogue-panel #send {width:34px !important;min-width:34px !important;max-width:34px !important;}
}
'''

IN_WORLD_CSS = '''
#game-world[data-dialogue="true"]::after {display:none;}
#dialogue-panel[data-game-open="true"] {display:flex !important;position:fixed;left:auto;right:20px;top:50%;transform:translateY(-50%);width:min(410px,43vw);max-height:calc(100dvh - 36px);padding:14px !important;gap:12px !important;overflow:visible !important;}
/* The speech stays viewport-anchored rather than inheriting the dock transform. */
#dialogue-panel[data-game-open="true"] {top:18px;transform:none;}
#dialogue-panel #content-grid {display:contents !important;}
#dialogue-panel #input-column,#dialogue-panel #player-replies {width:100% !important;max-width:none !important;flex:none !important;}
#dialogue-panel #conversation-column {position:fixed !important;left:var(--speech-left,24px);top:var(--speech-top,100px);width:var(--speech-width,320px) !important;max-width:none !important;min-width:0 !important;padding:0 !important;gap:6px !important;z-index:11;}
#dialogue-panel #mara-portrait {display:none !important;}
#dialogue-panel #conversation {height:190px !important;min-height:190px !important;max-height:190px !important;box-shadow:4px 4px 0 #51473680;}
#dialogue-panel #conversation::after {left:calc(50% - 10px);}
#dialogue-panel #activity {padding:5px 8px;background:#fff0d1;border:1px solid #b7a07a;}
#dialogue-panel #activity:empty {display:none;}
#dialogue-panel #live-camera {height:235px !important;min-height:235px !important;}
.dialogue-title {font-size:11px;gap:8px;}.dialogue-title span{max-width:16ch;}#close-dialogue{font-size:9px;padding:7px;}
@media(max-height:680px){
 #dialogue-panel #live-camera {height:170px !important;min-height:170px !important;}
 #dialogue-panel .sample-line {min-height:32px !important;padding:5px !important;}
}
@media(max-width:600px){
 #dialogue-panel[data-game-open="true"] {width:47vw;right:7px;top:10px;max-height:calc(100dvh - 20px);padding:7px !important;gap:7px !important;}
 .dialogue-title{flex-direction:column;align-items:flex-start;font-size:8px;gap:5px;}.dialogue-title span{max-width:none;}#close-dialogue{font-size:7px;padding:5px;}
 #dialogue-panel #live-camera{height:112px !important;min-height:112px !important;}
 #dialogue-panel #live-camera-tag{padding-top:3px;}
 #dialogue-panel #live-camera-tag .emotion-caption{font-size:7px;}
 #dialogue-panel .emotion-pill{font-size:8px;padding:3px;}
 #dialogue-panel #player-replies{padding-top:5px;gap:5px !important;}
 #dialogue-panel #player-replies .sample-row{gap:4px !important;}
 #dialogue-panel .sample-line{font-size:8px !important;min-height:28px !important;padding:4px !important;}
 #dialogue-panel #conversation {height:170px !important;min-height:170px !important;max-height:170px !important;}
 #dialogue-panel #activity p{font-size:7px !important;}
 #dialogue-panel #checkin-message textarea{font-size:8px !important;min-height:36px !important;height:36px;}
 #dialogue-panel #send{height:36px !important;min-height:36px !important;width:28px !important;min-width:28px !important;max-width:28px !important;}
}
'''

FLOATING_CSS = '''
#quest-event,#new-conversation {display:none !important;}
#example-note {background:#fff0d1 !important;border:2px solid #ad9368 !important;padding:6px 9px !important;}
#dialogue-panel #example-note p {font-size:10px !important;line-height:1.5 !important;color:#514332 !important;margin:0;}
@media(max-width:600px){#dialogue-panel #player-replies{max-height:calc(100dvh - 210px);overflow-y:auto !important;overflow-x:hidden !important;padding-right:3px;}#dialogue-panel #example-note p{font-size:8px !important;}}
#dialogue-panel[data-game-open="true"] {background:transparent !important;border:0 !important;box-shadow:none !important;border-radius:0 !important;padding:0 !important;right:18px;top:18px;gap:10px !important;}
#dialogue-header {background:transparent !important;}
#dialogue-header {flex:none !important;height:auto !important;}
#dialogue-header .html-container,#live-camera-tag .html-container {padding:0 !important;}
#dialogue-panel #input-column .tab-wrapper {display:none !important;}
#dialogue-panel #input-column .tabs,#dialogue-panel #input-column .tabitem > .column {gap:8px !important;}
.dialogue-title {border:0;padding:0;justify-content:flex-end;}
.dialogue-title > span {display:none;}
#close-dialogue {background:#fff0d1;border:2px solid #806b4d;box-shadow:2px 2px 0 #374431;font-size:11px;border-radius:0 !important;}
#dialogue-panel #input-column,#dialogue-panel #player-replies,#dialogue-panel #custom-reply-row {background:transparent !important;border:0 !important;box-shadow:none !important;}
#dialogue-panel #live-camera {border:3px solid #ebd4a0 !important;box-shadow:3px 3px 0 #384634 !important;}
#dialogue-panel #live-camera-tag {background:transparent !important;padding:5px 0 0;}
#dialogue-panel #live-camera-tag .emotion-caption {background:#fff0d1;padding:5px;color:#514332;}
#dialogue-panel #player-replies {padding-top:0;}
#dialogue-panel .sample-line {background:#fff0d1 !important;border:2px solid #ad9368 !important;box-shadow:2px 2px 0 #374431 !important;}
#dialogue-panel #checkin-message label.container > span {background:#fff0d1 !important;align-self:flex-start;padding:3px 6px !important;color:#514332 !important;}
#dialogue-panel #conversation::after {left:var(--speech-tail,calc(50% - 10px));}
#dialogue-panel #activity {background:transparent;border:0;text-shadow:1px 1px #223728;}
#dialogue-panel #activity p {color:#fff4d3 !important;}
#dialogue-panel[data-game-open="true"] {width:min(820px,58vw);}
#dialogue-panel #live-camera {height:clamp(280px,46vh,540px) !important;min-height:clamp(280px,46vh,540px) !important;}
/* A short stepped pixel bubble, anchored above the real world sprite. */
#dialogue-panel #conversation {height:130px !important;min-height:130px !important;max-height:130px !important;border:0 !important;border-radius:0 !important;background:#fff9e7 !important;box-shadow:0 -4px 0 #766047,0 4px 0 #766047,-4px 0 0 #766047,4px 0 0 #766047,4px 8px 0 #35443180 !important;}
#dialogue-panel #conversation::after {bottom:-20px;width:20px;height:20px;background:#766047;clip-path:polygon(0 0,100% 0,100% 40%,80% 40%,80% 60%,60% 60%,60% 80%,40% 80%,40% 100%,0 100%);}
#dialogue-panel #conversation::before {content:'';position:absolute;z-index:1;bottom:-12px;left:calc(var(--speech-tail,50%) + 4px);width:12px;height:16px;background:#fff9e7;clip-path:polygon(0 0,100% 0,100% 50%,67% 50%,67% 75%,34% 75%,34% 100%,0 100%);}
#dialogue-panel #conversation .message,#dialogue-panel #conversation .message *,#dialogue-panel .conversation-empty p {font-size:16px !important;line-height:1.5 !important;}
#dialogue-panel .sample-line {font-size:14px !important;min-height:52px !important;line-height:1.5;}
#dialogue-panel #checkin-message textarea {font-size:14px !important;height:52px;min-height:52px !important;padding:12px !important;}
#dialogue-panel #checkin-message label.container > span,#dialogue-panel #live-camera-tag .emotion-caption {font-size:11px !important;}
#dialogue-panel .emotion-pill {font-size:12px;}
#dialogue-panel #activity p {font-size:10px !important;}
#dialogue-panel #live-camera button {font-size:14px !important;}
#dialogue-panel #send {position:relative;display:block;width:52px !important;min-width:52px !important;max-width:52px !important;height:52px !important;min-height:52px !important;line-height:0 !important;}
#dialogue-panel #send::before {position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);margin:0 !important;}
#dialogue-panel .conversation-empty {max-width:none;text-align:left;margin:0;padding:12px;}
#dialogue-panel #conversation .bubble-wrap {overflow-y:auto !important;padding:8px !important;}
#dialogue-panel #activity {align-self:flex-start;padding:5px 0;margin-top:20px;}
@media(max-height:700px){#dialogue-panel #live-camera{height:30vh !important;min-height:150px !important;}}
@media(max-width:600px){#dialogue-panel[data-game-open="true"]{width:40vw;right:7px;top:10px;gap:6px !important;padding:0 !important;}.dialogue-title{align-items:flex-end;}}
@media(max-width:600px){#dialogue-panel #live-camera{height:150px !important;min-height:150px !important;}#dialogue-panel #conversation .message,#dialogue-panel #conversation .message *,#dialogue-panel .conversation-empty p{font-size:10px !important;}#dialogue-panel .conversation-empty{padding:4px;}#dialogue-panel .sample-line{font-size:9px !important;min-height:28px !important;}#dialogue-panel #checkin-message textarea{font-size:10px !important;height:36px;min-height:36px !important;padding:6px !important;}#dialogue-panel #send{width:32px !important;min-width:32px !important;max-width:32px !important;height:36px !important;min-height:36px !important;}#dialogue-panel #live-camera button{font-size:10px !important;}}
@media(max-width:600px) and (max-height:650px){
 #dialogue-panel #live-camera{height:clamp(84px,25vh,150px) !important;min-height:84px !important;}
 #dialogue-panel #live-camera-tag{min-height:0 !important;padding:0;}
 #dialogue-panel #live-camera-tag .emotion-caption{display:none;}
 #dialogue-panel #live-camera-tag .emotion-line{margin:0;}
 #dialogue-panel #custom-reply-row .form{min-width:0 !important;}
 #dialogue-panel #checkin-message label.container > span{font-size:8px !important;}
 #dialogue-panel #player-replies{gap:4px !important;}
 #dialogue-panel .sample-line{font-size:7px !important;line-height:1.4;min-height:28px !important;}
}
'''
