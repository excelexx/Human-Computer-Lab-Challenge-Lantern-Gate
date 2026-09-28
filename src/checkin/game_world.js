/* Pure movement/collision rules shared by the browser and Node regression tests. */
(function(root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.LanternWorld = api;
})(globalThis, () => {
  const WIDTH = 640, HEIGHT = 480;
  const NPC = Object.freeze({x:260,y:213});
  const buildings = [
    {x:240,y:80,w:160,h:116,kind:'gate'},
    {x:64,y:160,w:96,h:88,kind:'house'},
    {x:480,y:160,w:96,h:88,kind:'house'},
    {x:64,y:320,w:80,h:64,kind:'house'},
    {x:496,y:320,w:80,h:64,kind:'house'},
  ];
  const trees = [[40,58],[72,48],[116,70],[160,52],[202,54],[432,54],[476,68],[530,48],[584,68],
    [30,120],[30,190],[32,270],[38,342],[32,402],[604,120],[608,198],[604,266],[606,350],[602,406],
    [182,160],[458,160],[180,365],[458,365]];
  const solids = buildings.map(b=>({x:b.x,y:b.y+20,w:b.w,h:b.h-20})).concat(
    trees.map(([x,y])=>({x:x-5,y:y-5,w:10,h:10})),
    [{x:0,y:432,w:304,h:48},{x:336,y:432,w:304,h:48}, {x:NPC.x-7,y:NPC.y-9,w:14,h:12}]);
  const create = () => ({x:NPC.x,y:350,direction:'up',steps:0});
  const valid = (x,y) => x>=20 && x<=620 && y>=32 && y<=465 &&
    !solids.some(b=>x+5>b.x && x-5<b.x+b.w && y+3>b.y && y-3<b.y+b.h);
  function move(player, dx, dy, seconds) {
    const n=Math.hypot(dx,dy);
    if (!n || !Number.isFinite(seconds)) return false;
    const distance=90*Math.max(0,Math.min(seconds,.05));
    const x=player.x+dx/n*distance, y=player.y+dy/n*distance;
    let moved=false;
    if(x!==player.x&&valid(x,player.y)){player.x=x;moved=true;}
    if(y!==player.y&&valid(player.x,y)){player.y=y;moved=true;}
    if(moved) player.steps+=distance;
    player.direction=Math.abs(dx)>Math.abs(dy)?(dx>0?'right':'left'):(dy>0?'down':'up');
    return moved;
  }
  const near = p => Math.hypot(p.x-NPC.x,p.y-NPC.y)<37;
  return {WIDTH,HEIGHT,NPC,buildings,trees,solids,create,valid,move,near};
});
