"""Authored ambiguous lines and shared pixel-style interface accents."""

SAMPLE_LINES = (
    "Oh, fantastic.", "You want me to cross that?",
    "Sure. Whatever.", "What could possibly go wrong?",
)

PIXEL_CSS = '''
#masthead .brand-mark {border-radius:0; border:2px solid #e0be75; color:#e0be75; background:#243f53; font:22px Consolas,monospace;}
#masthead strong, #invitation h1, .panel-heading h2 {font-family:Consolas,'Courier New',monospace; font-weight:700; letter-spacing:-1px;}
#invitation {padding:22px 0 16px; align-items:center;}
#invitation h1 {font-size:clamp(28px,4vw,42px); line-height:1.15;}
#invitation p {max-width:49ch; font-size:14px;}
.pixel-scene {border:2px solid #597383; background:#14283d; margin-bottom:20px; box-shadow:5px 5px 0 #0d1e2c;}
.pixel-scene svg {display:block; width:100%; height:auto; image-rendering:pixelated;}
.scene-caption {display:flex; justify-content:space-between; gap:12px; padding:10px 14px; background:#20384b; color:#e0be75; font:12px Consolas,monospace;}
#input-column,#conversation-column {border-radius:2px; border:2px solid #3e596c; box-shadow:4px 4px 0 #0d1e2c;}
#send,#stop,#checkin-message textarea,#live-camera,#camera-clip,.emotion-pill {border-radius:2px !important;}
#send {background:#e0be75 !important; color:#172b3b !important; border-color:#e0be75 !important; box-shadow:3px 3px 0 #0d1e2c !important;}
#send:hover {background:#eed19a !important;}
#conversation .user,#conversation .bot {border-radius:2px !important;}
.conversation-empty {text-align:left; max-width:38ch;}
.conversation-empty h3 {color:#e0be75; font:700 21px Consolas,monospace;}
.conversation-empty .speaker {font:12px Consolas,monospace; color:#78c6c1; letter-spacing:2px; margin-bottom:18px;}
.sample-row {gap:7px !important;}
.sample-line {background:#203b50 !important; color:#dfe9ef !important; border:1px solid #526f82 !important; border-radius:2px !important; font-size:12px !important; text-align:left; min-height:35px !important; padding:8px 10px !important; box-shadow:none !important;}
.sample-line:hover {border-color:#e0be75 !important; color:#f1d69c !important;}
#sample-note p {font-size:12px; color:#a8bdcb; line-height:1.5;}
@media(max-width:600px){.scene-caption{flex-direction:column;gap:4px;font-size:11px;} .pixel-scene{margin-bottom:12px;} #invitation h1{font-size:30px;}}
'''
