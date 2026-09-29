() => {
  const key=Symbol.for('lanternGate.game');
  window[key]?.dispose();
  let disposed=false, raf=0, previous=0, open=false, autoArmed=true, hasMoved=false;
  const keys=new Set(), W=window.LanternWorld, player=W.create();
  let mara={...W.NPC},questPhase='talking',trip=null,readingLeft=0,lastSignal='',beaconLit=false;
  let reaction='idle',reactionToken='',reactionStarted=0,reactionAge=0;
  const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let canvas,ctx,world,panel,observer,help;
  const atlas=new Image();
  let pixelScale=2, viewW=480,viewH=320, camX=0,camY=0;
  const cleanups=[];
  function listen(target,event,handler,options){target.addEventListener(event,handler,options);cleanups.push(()=>target.removeEventListener(event,handler,options));}
  function cameraOff(){document.querySelector('#live-camera button[aria-label="Turn camera off"]')?.click();}
  function showHelp(){keys.clear();if(help&&!help.open){help.showModal();document.querySelector('#game-help-title').focus({preventScroll:true});help.scrollTop=0;}}
  function showDialogue(){
    if(open || !panel || questPhase!=='talking' || !W.near(player))return;
    open=true;keys.clear();world.inert=true;
    document.body.classList.add('in-dialogue');panel.inert=false;panel.dataset.gameOpen='true';world.dataset.dialogue='true';
    panel.setAttribute('aria-hidden','false');
    resize();
    document.querySelector('#close-dialogue')?.focus();
    document.querySelector('#game-hint').textContent='Talking with Mara. Escape returns to the village.';
  }
  function closeDialogue(pauseJourney=true){
    if(!open)return;
    document.querySelector('#leave-dialogue')?.click();
    cameraOff();open=false;autoArmed=false;keys.clear();world.inert=false;
    document.body.classList.remove('in-dialogue');panel.inert=true;panel.dataset.gameOpen='false';world.dataset.dialogue='false';
    panel.setAttribute('aria-hidden','true');resize();canvas.focus();
    if(pauseJourney && questPhase==='reading')questPhase='paused';
  }
  function resize(){
    pixelScale=Math.max(1,Math.floor(Math.min(innerWidth/400,innerHeight/270)));
    viewW=Math.floor(innerWidth/pixelScale);viewH=Math.floor(innerHeight/pixelScale);
    canvas.width=viewW;canvas.height=viewH;ctx.imageSmoothingEnabled=false;
    canvas.style.width=`${viewW*pixelScale}px`;canvas.style.height=`${viewH*pixelScale}px`;
  }
  const isTyping=target=>target?.closest?.('input,textarea,select,[contenteditable="true"]');
  function keydown(event){
    if(help?.open)return; // Native dialog owns Escape and keyboard focus.
    if(open){
      if(event.key==='Escape'){event.preventDefault();closeDialogue();}
      if(event.key==='Tab'){
        const focusable=[...panel.querySelectorAll('button,input,textarea,select,[tabindex="0"]')].filter(el=>!el.disabled&&el.getClientRects().length);
        const first=focusable[0],last=focusable.at(-1);
        if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}
        else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}
      }
      return;
    }
    if(questPhase==='walking'||questPhase==='paused'){
      if(event.key==='Escape'){event.preventDefault();questPhase='paused';}
      if(event.key.toLowerCase()==='e'){event.preventDefault();questPhase='walking';}
      return;
    }
    if(isTyping(event.target))return;
    // Keep native Enter/Space activation on HUD buttons accessible.
    if(event.target?.closest?.('button') && ['Enter',' '].includes(event.key))return;
    const value=event.key.toLowerCase();
    if(['w','a','s','d','arrowup','arrowdown','arrowleft','arrowright'].includes(value)){
      event.preventDefault();keys.add(value);
    }
    if(value==='e'||event.key==='Enter'){event.preventDefault();showDialogue();}
  }
  function rect(x,y,w,h,color){ctx.fillStyle=color;ctx.fillRect(Math.round(x),Math.round(y),w,h);}
  function tile(col,row,x,y,w=16,h=16){if(atlas.complete&&atlas.naturalWidth)ctx.drawImage(atlas,col*16,row*16,16,16,Math.round(x),Math.round(y),w,h);}
  function label(text,x,y,color='#fff0c2'){
    ctx.font='8px PixelTown, monospace';ctx.textAlign='center';
    ctx.fillStyle='#25313b';ctx.fillText(text,Math.round(x)+1,Math.round(y)+1);
    ctx.fillStyle=color;ctx.fillText(text,Math.round(x),Math.round(y));
  }
  function building(b){
    // Tile walls, roof, windows and a doorway from the CC0 town atlas.
    rect(b.x+5,b.y+8,b.w,b.h,'#537753');
    for(let y=b.y+32;y<b.y+b.h;y+=16)for(let x=b.x;x<b.x+b.w;x+=16)tile(b.kind==='gate'?5:1,6,x,y);
    for(let y=b.y;y<b.y+48;y+=16)for(let x=b.x;x<b.x+b.w;x+=16)tile(b.kind==='gate'?1:5,y===b.y?4:5,x,y);
    rect(b.x,b.y+45,b.w,4,'#4d3e43');
    for(let x=b.x+12;x<b.x+b.w-16;x+=32){tile(b.kind==='gate'?4:0,7,x,b.y+57);}
    const door=b.x+b.w/2-8;
    tile(b.kind==='gate'?6:2,6,door,b.y+b.h-32);tile(b.kind==='gate'?5:1,7,door,b.y+b.h-16);
    rect(door-3,b.y+b.h,22,4,'#e6d6b6');
    if(b.kind==='gate'){
      for(const x of [b.x-12,b.x+b.w-12]){
        rect(x,b.y+10,24,b.h-6,'#65737e');rect(x+3,b.y+13,18,b.h-10,'#bcc0bd');
        for(let y=b.y+18;y<b.y+b.h;y+=12){rect(x+3,y,18,1,'#8e9698');rect(x+11+(y%24?0:5),y,1,11,'#8e9698');}
        rect(x-3,b.y+5,30,6,'#65737e');
        for(let i=0;i<3;i++)rect(x-3+i*12,b.y,6,8,'#bcc0bd');
      }
      label('Lantern Gate',b.x+b.w/2,b.y+70);
      for(const x of [b.x+25,b.x+b.w-33]){rect(x,b.y+73,7,27,'#ab5549');rect(x+2,b.y+77,3,10,'#eac879');}
    }
  }
  function person(x,y,kind,moving=false){
    x=Math.round(x);y=Math.round(y);
    rect(x-7,y-1,14,4,'#557553');
    const bob=moving&&!reduce?Math.floor(player.steps/7)%2:0;
    const pose=kind==='mara'?reaction:'idle';
    const nod=!reduce&&['steady','practical'].includes(pose)&&reactionAge<900?Math.floor(reactionAge/225)%2:0;
    const top=y-22-bob+nod, face=kind==='mara'?'#e7af7d':'#e8bb8b';
    const coat=kind==='mara'?'#56856d':'#5d85bd', shade=kind==='mara'?'#36554e':'#3e577c';
    rect(x-5,top,10,3,kind==='mara'?'#b87d49':'#493e3d');
    rect(x-7,top+3,14,7,kind==='mara'?'#b87d49':'#493e3d');
    rect(x-5,top+4,10,8,face);rect(x-3,top+7,2,2,'#31313a');rect(x+2,top+7,2,2,'#31313a');
    rect(x-6,top+12,12,8,coat);rect(x-7,top+17,14,3,shade);
    if(!['careful','patient'].includes(pose))rect(x-9,top+13,3,6,face);
    if(pose!=='playful')rect(x+6,top+13,3,6,face);
    rect(x-5,y-3-bob,4,5,'#3a3440');rect(x+1,y-3+bob,4,5,'#3a3440');
    if(kind==='mara'){
      const lift=pose==='playful'?5+(!reduce&&reactionAge<1200?Math.floor(reactionAge/200)%2:0):0;
      if(lift){rect(x+6,top+11,3,4,face);rect(x+8,top+9,4,3,face);}
      rect(x+10,top+14-lift,5,8,'#775545');rect(x+11,top+15-lift,3,5,'#ffe1a0');
      rect(x-5,top+12,10,2,'#eac879');
      if(pose==='playful'){
        rect(x-3,top+9,1,1,'#704a37');rect(x+2,top+9,1,1,'#704a37');rect(x-2,top+10,4,1,'#704a37');
        if(reactionAge<1200){rect(x+17,top+4,1,5,'#fff2bd');rect(x+15,top+6,5,1,'#fff2bd');}
      }else if(pose==='careful'){
        rect(x-9,top+14,3,3,face);rect(x-13,top+13,5,2,face);rect(x-14,top+11,1,2,face);
        rect(x-4,top+5,2,1,'#775545');rect(x+2,top+5,2,1,'#775545');rect(x-1,top+10,3,1,'#986144');
      }else if(pose==='patient'){
        rect(x-8,top+13,3,4,face);rect(x-5,top+14,5,2,face);rect(x-1,top+13,2,3,face);
        rect(x-3,top+7,3,1,'#31313a');rect(x+2,top+7,3,1,'#31313a');rect(x-1,top+10,2,1,'#986144');
      }else if(pose==='steady'){
        rect(x-4,top+5,3,1,'#775545');rect(x+1,top+5,3,1,'#775545');rect(x-2,top+10,4,1,'#704a37');
      }else if(pose==='curious'){
        rect(x-4,top+4,3,1,'#775545');rect(x+2,top+5,2,1,'#775545');rect(x,top+9,2,2,'#704a37');
      }else if(pose==='wry'){
        rect(x-4,top+4,3,1,'#775545');rect(x-1,top+10,3,1,'#704a37');rect(x+2,top+9,1,1,'#704a37');
      }else if(pose==='practical')rect(x-1,top+10,3,1,'#986144');
    }else{
      rect(x-5,top+11,10,2,'#f0d394');
      if(player.direction==='up'){rect(x-5,top+4,10,6,'#493e3d');rect(x-4,top+14,8,5,'#bc8c60');}
    }
  }
  function movementHint(time){
    if(open||questPhase!=='talking'||W.near(player))return;
    const x=Math.round(player.x), y=Math.round(player.y);
    const bob=reduce?0:Math.floor(time/350)%2;
    // A large, outlined direction arrow above the traveler points toward Mara.
    ctx.save();ctx.translate(x,y-(hasMoved?54:108)+bob);
    const angle=Math.round((Math.atan2(W.NPC.y-player.y,W.NPC.x-player.x)+Math.PI/2)/(Math.PI/2))*Math.PI/2;
    ctx.rotate(angle);
    // Axis-aligned pixel steps rather than a rasterized diagonal path.
    for(let row=0;row<8;row++){const half=Math.min(row,6);rect(-half*2-2,-16+row*2,half*4+4,2,'#5c4737');}
    rect(-6,0,12,16,'#5c4737');
    for(let row=1;row<7;row++){const half=Math.max(0,row-1);rect(-half*2-1,-16+row*2,half*4+2,2,'#ffe39a');}
    rect(-4,-2,8,16,'#ffe39a');ctx.restore();
    if(hasMoved)return;
    label('Move up',x,y-83);
    for(const [key,dx,dy] of [['W',0,-70],['A',-17,-53],['S',0,-53],['D',17,-53]]){
      rect(x+dx-7,y+dy-7,15,15,'#50483e');rect(x+dx-6,y+dy-6,13,13,key==='W'?'#ffe39a':'#fff0cc');
      const letters={W:['10001','10001','10101','10101','10101','11011','10001'],A:['01110','10001','10001','11111','10001','10001','10001'],S:['01111','10000','10000','01110','00001','00001','11110'],D:['11110','10001','10001','10001','10001','10001','11110']};
      letters[key].forEach((row,iy)=>[...row].forEach((v,ix)=>{if(v==='1')rect(x+dx-2+ix,y+dy-3+iy,1,1,'#4c4236');}));
    }
  }
  function draw(moving,time){
    ctx.imageSmoothingEnabled=false;
    // Keep the traveler and Mara in the left half during both walking and talking.
    // This framing is independent of dialogue state, so opening never shifts it.
    camX=Math.round(Math.max(Math.min(player.x-viewW*.28,W.WIDTH-viewW),Math.min(0,(W.WIDTH-viewW)/2)));
    camY=Math.round(Math.max(Math.min(player.y-viewH*.6,W.HEIGHT-viewH),Math.min(0,(W.HEIGHT-viewH)/2)));
    if(open){
      // Opening dialogue never pans or zooms the world. Anchor the bubble to
      // Mara's actual screen position, keeping its right edge clear of controls.
      const dock=panel.getBoundingClientRect(), available=Math.max(100,dock.left-16);
      const speech=document.querySelector('#conversation-column');
      const width=Math.min(460,available-16), npcY=(W.NPC.y-camY)*pixelScale;
      const npcX=(W.NPC.x-camX)*pixelScale;
      const left=Math.max(8,Math.min(npcX-width/2,available-width));
      speech.style.setProperty('--speech-left',`${left}px`);
      speech.style.setProperty('--speech-width',`${width}px`);
      const cueHeight=document.querySelector('#reply-cue')?.getBoundingClientRect().height||0;
      const bubbleBottom=npcY-22*pixelScale-26; // Leave room for the tail above Mara's head.
      const speechHeight=document.querySelector('#conversation').getBoundingClientRect().height;
      speech.style.setProperty('--speech-top',`${Math.max(12,bubbleBottom-speechHeight-cueHeight-6)}px`);
      speech.style.setProperty('--speech-tail',`${Math.max(14,Math.min(width-26,npcX-left-10))}px`);
    }
    rect(0,0,viewW,viewH,'#243e4a');ctx.save();ctx.translate(-camX,-camY);
    rect(0,0,W.WIDTH,W.HEIGHT,'#86ad69');
    for(let y=0;y<432;y+=16)for(let x=0;x<640;x+=16){
      const noise=(x*31+y*17)%101;
      tile(noise<13?1:0,0,x,y);
    }
    // Sandstone plaza and linking paths. Brick joints are offset row by row.
    const paths=[{x:208,y:192,w:224,h:240},{x:144,y:256,w:352,h:64}];
    for(const p of paths){
      rect(p.x-3,p.y-3,p.w+6,p.h+6,'#697d68');rect(p.x,p.y,p.w,p.h,'#b5b4aa');
      for(let y=p.y;y<p.y+p.h;y+=8){
        rect(p.x,y,p.w,1,'#90978f');
        for(let x=p.x+((y/8)%2)*8;x<p.x+p.w;x+=16)rect(x,y,1,8,'#90978f');
      }
    }
    rect(0,432,640,48,'#477888');rect(0,428,640,4,'#d1d0b1');
    for(let y=443;y<480;y+=13)for(let x=10;x<640;x+=38){rect(x+(reduce?0:Math.floor(time/700)%3),y,15,1,'#70a8b0');}
    rect(304,415,32,65,'#8a6850');
    for(let y=416;y<480;y+=7){rect(304,y,32,2,'#5e4b3d');rect(306,y+2,28,1,'#bfa177');}
    // A second, visibly distinct route: sheltered stone steps to the beacon pier.
    rect(264,394,24,69,'#627a79');
    for(let y=396;y<460;y+=6){rect(265,y,22,4,'#bcc0af');rect(267,y,18,1,'#e0dfc5');}
    rect(264,448,88,32,'#8a6850');
    for(let y=450;y<480;y+=6)rect(264,y,88,2,'#bc946b');
    for(const [x,y] of [[191,249],[447,249],[191,334],[447,334]]){
      rect(x-2,y-27,4,29,'#414b4a');rect(x-5,y-31,10,9,'#60574a');rect(x-3,y-29,6,5,'#f3d591');
    }
    // Small garden beds, flowers, crates and benches make the courtyard lived in.
    for(const x of [168,440])for(const y of [209,229,386]){tile(4,2,x,y);tile(2,0,x+16,y);}
    for(const [x,y] of [[172,279],[449,279],[82,263],[546,263]]){tile(10,6,x,y);tile(11,6,x+16,y);}
    const objects=W.buildings.map(b=>({y:b.y+b.h,draw:()=>building(b)}));
    for(const [x,y] of W.trees)objects.push({y,draw:()=>{rect(x-9,y-1,20,4,'#557753');tile(4,0,x-12,y-48,24,24);tile(4,1,x-12,y-24,24,24);}});
    objects.push({y:mara.y,draw:()=>person(mara.x,mara.y,'mara',questPhase==='walking')});
    objects.push({y:player.y,draw:()=>person(player.x,player.y,'player',moving)});
    objects.sort((a,b)=>a.y-b.y).forEach(o=>o.draw());
    objects.push({y:466,draw:()=>{
      rect(333,445,14,23,'#596167');rect(336,442,8,24,'#c7c3ab');
      rect(331,433,18,12,'#4a4542');rect(334,435,12,8,beaconLit?'#ffe18a':'#6b776b');
      rect(329,431,22,3,'#665044');rect(331,468,18,3,'#3d4c49');
      if(beaconLit){rect(338,438,4,3,'#fffce1');rect(337,424,6,4,'#fff1a9');rect(322,434,5,3,'#f8d67f');rect(353,434,5,3,'#f8d67f');}
    }});
    // Draw the beacon after the pier, with a visible extinguished/lit state.
    objects.at(-1).draw();
    label('Mara',mara.x,mara.y-29);
    if(!open && questPhase==='talking'){
      const bob=reduce?0:Math.floor(time/400)%2;
      rect(W.NPC.x-1,W.NPC.y-47+bob,3,7,'#ffe5a4');rect(W.NPC.x-1,W.NPC.y-38+bob,3,2,'#ffe5a4');
      if(W.near(player))label('E · Talk',W.NPC.x,W.NPC.y+16);
    }
    movementHint(time);ctx.restore();
  }
  function tick(time){
    if(disposed)return;
    const delta=previous?Math.min((time-previous)/1000,.05):0;previous=time;
    const dx=(keys.has('d')||keys.has('arrowright')?1:0)-(keys.has('a')||keys.has('arrowleft')?1:0);
    const dy=(keys.has('s')||keys.has('arrowdown')?1:0)-(keys.has('w')||keys.has('arrowup')?1:0);
    // This signal comes only from the response turn, never the changing webcam tag.
    const cue=document.querySelector('#reply-cue-signal');
    const nextReaction=['playful','careful','steady','patient','curious','wry','practical'].includes(cue?.dataset.reaction)?cue.dataset.reaction:'idle';
    const nextToken=(cue?.dataset.turn||'')+':'+nextReaction;
    if(nextToken!==reactionToken){reactionToken=nextToken;reaction=nextReaction;reactionStarted=time;}
    reactionAge=time-reactionStarted;
    const signal=document.querySelector('#quest-signal');
    if(signal){
      const s=signal.dataset, token=[s.session,s.count,s.route,s.phase].join(':');
      if(token!==lastSignal){
        lastSignal=token;
        if(s.phase==='depart' && ['bridge','stairs'].includes(s.route) && questPhase==='talking'){
          trip=W.journey(s.route,player);questPhase='reading';readingLeft=6;keys.clear();
        }else if(s.phase==='talking' && s.count==='0' && questPhase!=='talking'){
          Object.assign(player,W.create());mara={...W.NPC};trip=null;beaconLit=false;questPhase='talking';autoArmed=true;hasMoved=false;
        }
      }
    }
    if(questPhase==='reading' && !document.hidden && !help?.open){
      readingLeft-=delta;
      if(readingLeft<=0){closeDialogue(false);questPhase='walking';}
    }
    if(questPhase==='walking' && !document.hidden && !help?.open){
      if(reduce)for(let i=0;i<1000&&!trip.done;i++)W.advanceJourney(trip,.05);
      else W.advanceJourney(trip,delta);
      mara={...trip.mara};Object.assign(player,trip.traveler);player.steps+=58*delta;
      player.direction='down';
      if(trip.done){questPhase='complete';beaconLit=true;}
    }
    const moving=!open&&!help?.open&&['talking','complete'].includes(questPhase)&&W.move(player,dx,dy,delta);
    if(moving)hasMoved=true;
    const near=W.near(player);
    if(!near)autoArmed=true;
    if(near&&autoArmed&&!open&&!help?.open&&questPhase==='talking'){autoArmed=false;showDialogue();}
    document.querySelector('#restart-scene').hidden=questPhase==='talking';
    if(!open)document.querySelector('#game-hint').textContent=questPhase==='complete'?'Beacon restored! Mara led you along the '+(trip.route==='bridge'?'signal bridge.':'sea stairs.')+' Explore with WASD or restart the scene.':questPhase==='paused'?'Journey paused. Press E to continue or restart the scene.':questPhase==='walking'?'Following Mara along the '+(trip.route==='bridge'?'signal bridge':'sea stairs')+'. Esc pauses the journey.':near?'Mara is here. Press E to talk.':'Find Mara at the northern gate. Walk up the stone path.';
    canvas.dataset.questPhase=questPhase;canvas.dataset.beaconLit=String(beaconLit);canvas.dataset.route=trip?.route||'';
    canvas.dataset.maraX=mara.x.toFixed(1);canvas.dataset.maraY=mara.y.toFixed(1);
    canvas.dataset.maraReaction=reaction;
    canvas.dataset.playerX=player.x.toFixed(1);canvas.dataset.playerY=player.y.toFixed(1);
    canvas.dataset.nearMara=String(near);canvas.dataset.dialogueOpen=String(open);
    canvas.dataset.movementHint=String(!hasMoved);
    draw(moving,time);raf=requestAnimationFrame(tick);
    canvas.dataset.viewX=String(camX);canvas.dataset.viewY=String(camY);canvas.dataset.pixelScale=String(pixelScale);
  }
  function attach(){
    canvas=document.querySelector('#town-canvas');panel=document.querySelector('#dialogue-panel');
    if(!canvas||!panel||!document.querySelector('#close-dialogue'))return;
    observer?.disconnect();ctx=canvas.getContext('2d');world=document.querySelector('#game-world');
    help=document.querySelector('#game-help');
    atlas.src=canvas.dataset.atlas;
    document.body.classList.add('game-ready');document.body.style.overflow='hidden';panel.dataset.gameOpen='false';world.dataset.dialogue='false';panel.inert=true;panel.setAttribute('aria-hidden','true');
    panel.setAttribute('role','dialog');panel.setAttribute('aria-modal','true');panel.setAttribute('aria-label','Speak with Mara');
    listen(window,'resize',resize);listen(window,'keydown',keydown);listen(window,'keyup',e=>keys.delete(e.key.toLowerCase()));
    listen(window,'blur',()=>keys.clear());
    listen(document,'visibilitychange',()=>{keys.clear();if(document.hidden)cameraOff();});
    listen(document.querySelector('#close-dialogue'),'click',()=>closeDialogue());
    for(const button of document.querySelectorAll('[data-game-help]'))listen(button,'click',showHelp);
    listen(help,'close',()=>{keys.clear();try{sessionStorage.setItem('lanternGate.helpSeen.v1','yes');}catch{}if(!open)canvas.focus();});
    listen(document.querySelector('#restart-scene'),'click',()=>{cameraOff();document.querySelector('#new-conversation')?.click();});
    listen(document.querySelector('#game-fullscreen'),'click',async()=>{
      try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}
      catch{document.querySelector('#game-hint').textContent='Browser full screen is unavailable. The game still fills this tab.';}
    });
    resize();raf=requestAnimationFrame(tick);
    try{if(sessionStorage.getItem('lanternGate.helpSeen.v1')!=='yes')showHelp();}catch{showHelp();}
  }
  observer=new MutationObserver(attach);observer.observe(document.documentElement,{childList:true,subtree:true});attach();
  window[key]={dispose(){disposed=true;cancelAnimationFrame(raf);observer.disconnect();cleanups.forEach(f=>f());keys.clear();cameraOff();document.body.classList.remove('game-ready','in-dialogue');}};
  listen(window,'pagehide',()=>window[key]?.dispose(),{once:true});
}
