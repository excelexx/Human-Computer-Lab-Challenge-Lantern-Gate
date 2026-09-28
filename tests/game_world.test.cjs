const test=require('node:test'),assert=require('node:assert/strict');
const W=require('../src/checkin/game_world.js');
test('WASD path from spawn reaches Mara without crossing a wall',()=>{
 const p=W.create();assert(!W.near(p));
 for(let i=0;i<90&&!W.near(p);i++)W.move(p,0,-1,1/60);
 assert(W.near(p));assert(W.valid(p.x,p.y));assert.equal(p.direction,'up');
});
test('walking cannot tunnel into the gate or the sea, even after a long suspended frame',()=>{
 const p=W.create();for(let i=0;i<200;i++)W.move(p,0,-1,10);
 assert(p.y>=219);assert(W.valid(p.x,p.y));
 const q={...W.create(),x:260,y:425};for(let i=0;i<200;i++)W.move(q,0,1,10);
 assert(q.y<=429);assert(W.valid(q.x,q.y));
});
test('diagonal movement has the same speed as cardinal movement',()=>{
 const a={...W.create(),x:320,y:330},b={...W.create(),x:320,y:330};
 W.move(a,1,0,.04);W.move(b,1,1,.04);
 assert(Math.abs(Math.hypot(a.x-320,a.y-330)-Math.hypot(b.x-320,b.y-330))<1e-8);
});
test('no keys, negative delta and invalid time cannot move the player',()=>{
 const p=W.create(),before={...p};W.move(p,0,0,.05);W.move(p,1,0,NaN);
 assert.deepEqual(p,before);W.move(p,1,0,-10);assert.equal(p.x,before.x);
});
test('world and buildings bound all solid collision rectangles',()=>{
 assert(W.solids.length>20);assert(!W.valid(5,5));assert(!W.valid(320,100));
 assert(W.valid(320,450));assert(!W.valid(280,450));
});
test('onboarding remains until there is actual displacement',()=>{
 const p={...W.create(),y:219};
 assert.equal(W.move(p,0,-1,.05),false);
 assert.equal(W.move(p,0,1,0),false);
 assert.equal(W.move(p,0,1,.05),true);
});
