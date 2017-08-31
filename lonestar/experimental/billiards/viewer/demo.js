function assert(expression, msg) {  
   if (!expression) {  
      if (msg === undefined) {
         console.log("Assertion failed.");
      } else {
         console.log("Assertion failed: " + msg);
      }
   }
};

Config = function (length, width, numBalls, mass, radius) {
  this.length = length;
  this.width = width;
  this.numBalls = numBalls;
  this.mass = mass;
  this.radius = radius;
};

Config.prototype = {
  constructor: Config,
};


// TODO: add delta field to ball
Ball = function (id, pos, vel, radius) {
  assert (pos instanceof THREE.Vector2);
  assert (vel instanceof THREE.Vector2);

  this.id = id;
  this.pos = pos; // new THREE.Vector2 (0.0, 0.0);
  this.vel = vel; // new THREE.Vector2 (0.0, 0.0);
  this.time = 0.0;
  this.disp = new THREE.Vector2 (0.0, 0.0); // temporary vector to hold v*dt

  var red = 0xff * (0.3 + 0.7 * Math.random ());
  var green = 0xff * (0.3 + 0.7 * Math.random ());
  var blue = 0xff * (0.3 + 0.7 * Math.random ());
  this.color = new THREE.Color ((red << 16) | (green << 8) | blue);
  this.brightColor = new THREE.Color (0xff0000);

  var sphereMaterial = new THREE.MeshPhongMaterial ({
    color: this.color,
    shininess: 50, 
    shading: THREE.SmoothShading
  });

  var NLAT = 16;
  var NLONG = 16;
  this.sphereGeom = new THREE.SphereGeometry (radius, NLAT, NLONG);
  this.sphere = new THREE.Mesh (this.sphereGeom, sphereMaterial);

  this.lineGeom = new THREE.Geometry ();
  this.lineGeom.vertices.push (new THREE.Vector3 (0, 0, 0));
  this.lineGeom.vertices.push (new THREE.Vector3 (vel.x, vel.y, 0.0));
  this.lineGeom.verticesNeedUpdate = true;
  this.lineGeom.dynamic = true;
  this.line =  new THREE.Line (this.lineGeom, new THREE.LineBasicMaterial ({ color:0xffffff, linewidth: 1}));

};

Ball.prototype = {
  constructor: Ball,

  advanceBy: function (delta) {
    // this.pos += this.vel * (delta);
    this.disp.copy (this.vel);
    this.disp.multiplyScalar (delta);
    this.pos.add (this.disp);
    this.time += delta;
    this.updateShapePos ();
  },

  advanceUpto: function (endTime) {
    assert (endTime >= this.time, "can't advance back in time");
    this.advanceBy (endTime - this.time);
  },

  applyUpdate: function (up) {
    // assert (up.time >= this.time, "can't advance back in time");
    assert (up.id == this.id, "id mismatch on update and ball");

    this.pos.copy (up.pos);
    this.vel.copy (up.vel);
    this.time = up.time;

    this.updateShapePos ();
    this.updateShapeVel ();
  },

  updateShapePos: function () {
    this.sphere.position.x = this.pos.x;
    this.sphere.position.y = this.pos.y;
    this.line.position.copy (this.sphere.position);
  },

  updateShapeVel: function () {
    // this.lineGeom.vertices[1].copy (this.vel);
    this.lineGeom.vertices[1].x = this.vel.x;
    this.lineGeom.vertices[1].y = this.vel.y;
    this.line.geometry.verticesNeedUpdate = true;
    this.line.geometry.dynamic = true;
  },

  addToScene: function (scene) {
    this.updateShapePos ();
    this.updateShapeVel ();
    scene.add (this.sphere);
    scene.add (this.line);
  },

  brighten: function () {
    this.sphere.material.color = this.brightColor;
  },

  restoreColor: function () {
    this.sphere.material.color = this.color;
  },
};

BallUpdate = function (id, pos, vel, time) {
  assert (pos instanceof THREE.Vector2);
  assert (vel instanceof THREE.Vector2);
  this.id = id;
  this.pos = pos;
  this.vel = vel;
  this.time = time;
  this.delta = 0.0;
};

BallUpdate.prototype = {
  constructor: BallUpdate
};

updateComparator = function (up1, up2) {
  return up1.time - up2.time;
};

UpdateStep = function (id) {
  this.id = id;
  this.updates = [];
};

UpdateStep.prototype = {
  constructor: UpdateStep,

  addUpdate: function (up) {
    this.updates.push (up);
  },

  sort: function () {
    this.updates.sort (updateComparator);
  },
};


ArrayIterator = function (array) {
  this.index = 0;
  this.array = array;
};

ArrayIterator.prototype = {
  constructor: ArrayIterator,

  hasMore: function () {
    return this.index < this.array.length;
  },

  get: function () {
    return this.array[this.index];
  },

  increment: function () {
    ++this.index;
  }
};

readCSVfile = function (fname) {
  var lines = [];
  var tokens = [];
  var file = new XMLHttpRequest ();
  file.overrideMimeType ('text/plain');
  file.open ('GET', fname, false);
  file.send ();

  lines = file.responseText.split ("\n"); // Will separate each line into an array
  if (lines[lines.length - 1] == "") { lines.pop (); }

  // TODO: split each line into words by comma
  //
  for (var l in lines) {
    var tok = lines[l].split (",");
    tokens.push (tok);
  }

  return tokens;
};

readConfig = function (configFile) {
  var tok = readCSVfile (configFile);
  var config = new Config (tok[1][0], tok[1][1], tok[1][2], tok[1][3], tok[1][4]);
  return config;
};

readInitState = function (config, balls, ballsFile) {
  var tokens = readCSVfile (ballsFile);

  //console.log (tokens.length);
  //console.log (config.numBalls);
  assert (tokens.length - 1 == config.numBalls, "initial state token count error");
  

  for (var row = 1; row < tokens.length; ++row) { // ignore header row
    var id = parseInt (tokens[row][0]);
    var pos = new THREE.Vector2 (parseFloat (tokens[row][1]), parseFloat (tokens[row][2]), 0.0);
    var vel = new THREE.Vector2 (parseFloat (tokens[row][3]), parseFloat (tokens[row][4]), 0.0);

    var b = new Ball (id, pos, vel, config.radius);
    balls.push (b);
  }
};

readSimLog = function (updateSteps, simLogFile) {

  var tokens = readCSVfile (simLogFile);
  var maxStepID = 0;

  for (var row = 1; row < tokens.length; ++row) { // ignore header row
    var stepID = parseInt (tokens[row][0]);
    var time = parseFloat (tokens[row][1]);
    var ballID = parseInt (tokens[row][2]);
    var pos = new THREE.Vector2 (parseFloat (tokens[row][3]), parseFloat (tokens[row][4])); 
    var vel = new THREE.Vector2 (parseFloat (tokens[row][5]), parseFloat (tokens[row][6]));
    var up = new BallUpdate (ballID, pos, vel, time);
    maxStepID = Math.max(maxStepID, stepID);

    if (updateSteps[stepID] == undefined) {
      updateSteps[stepID] = new UpdateStep (stepID);
    }
    updateSteps[stepID].updates.push (up);
  }

  for (var i = 0; i < maxStepID; ++i) {
    if (updateSteps[i] === undefined) {
      updateSteps[i] = new UpdateStep(i);
    }
  }
};


var config;
var balls = [];
var updateSteps = [];
var updateStepsIterator;
var endTime = 0;
var deltaT = 0.01;
var timer = 0;
var reinitScene;

populateScene = function (scene, configFile, ballsFile, simLogFile) {
  config = undefined;
  balls = [];
  updateSteps = [];
  updateStepsIterator = undefined;
  endTime = 0;
  deltaT = 0.01;
  timer = getTime ();

  config = readConfig (configFile);

  // create borderes
  var borderGeom = new THREE.Geometry ();
  borderGeom.vertices.push (new THREE.Vector3 (0, 0, 0));
  borderGeom.vertices.push (new THREE.Vector3 (config.length, 0, 0));
  borderGeom.vertices.push (new THREE.Vector3 (config.length, config.width, 0));
  borderGeom.vertices.push (new THREE.Vector3 (0, config.width, 0));
  borderGeom.vertices.push (new THREE.Vector3 (0, 0, 0));

  var border = new THREE.Line (borderGeom,
      new THREE.LineBasicMaterial ({ color: 0xffffff, linewidth: 2}));

  scene.add (border);

  readInitState (config, balls, ballsFile);
  readSimLog (updateSteps, simLogFile);
  assert (updateSteps.length > 0, "no updates read?");

  for (var b in balls) {
    balls[b].addToScene (scene);
  }

  // console.log (balls);
  updateStepsIterator = new ArrayIterator (updateSteps);
};

// steps
// - setup camera lighting
// - create scene
// - read configuration
// - read initial state
// - add balls to the scene
// - read updates
// - animate

var scene;
var camera;
var renderer;

function initRenderer (docElement) {
  // renderer = new THREE.CanvasRenderer();
  renderer = new THREE.WebGLRenderer();
  renderer.setSize( window.innerWidth, window.innerHeight);

  docElement.appendChild (renderer.domElement);
}

function initScene (configFile, ballsFile, simLogFile) {

  renderer.clear ();
  window.cancelAnimationFrame (animFrame);
  scene = undefined;
  camera = undefined;

  // camera = new THREE.PerspectiveCamera( 1000, window.innerWidth / window.innerHeight, 1, 1000 );
  // camera.position.set (0,0, 1000);
  // camera.lookAt (scene.position);

  // camera = new THREE.OrthographicCamera (-window.innerWidth/10, window.innerWidth/10, -window.innerHeight/10, window.innerHeight/10);

  //camera = new THREE.OrthographicCamera (0, window.innerWidth/8, 0, window.innerHeight/8);
  camera = new THREE.OrthographicCamera (0, window.innerWidth/3, 0, window.innerHeight/3);
  camera.position.set(0, 0, 100);
  camera.lookAt(new THREE.Vector3(0, 0, 0));

  scene = new THREE.Scene();
  scene.add (camera);

  // scene.add (new THREE.AmbientLight (0x111111));

  // create a point light
  // var light = new THREE.PointLight(0xFFFFFF);
  var light = new THREE.DirectionalLight (0xFFFFFF);

  // set its position
  light.position.x = 0;
  light.position.y = 0;
  light.position.z = -1000;

  // add to the scene
  scene.add(light);

  populateScene (scene, configFile, ballsFile, simLogFile);
}

function getTime () { return Date.now () * 0.0005; }

var animFrame = 0;
function animate() {

  // note: three.js includes requestAnimationFrame shim
  animFrame = requestAnimationFrame (animate);

  if (getTime () - timer < deltaT) { return; }
  timer = getTime ();
  endTime += deltaT;
  // console.log (endTime);

  while (updateStepsIterator.hasMore ()) {
    var upStep = updateStepsIterator.get ();

    var allDone = true;
    for (var i in upStep.updates) {
      var up = upStep.updates[i];

      if (endTime > up.time) { 
        // up should be applied now
        balls[up.id].applyUpdate (up);
      } else { 
        // this step still has updates that need to be applied
        allDone = false; 
      }
    }

    if (allDone) { 
      updateStepsIterator.increment ();
    } else { 
      break;
    }
  }

  //reinitScene();

  for (var b in balls) {
    balls[b].advanceUpto (endTime);
  }

  renderer.render (scene, camera);
}

var NDELTA = 50;
function animateEvents () {
  animFrame = requestAnimationFrame (animateEvents);

  // if (getTime () - timer < deltaT) { return; }
  timer = getTime ();

  if (updateStepsIterator.hasMore ()) {
    var upStep = updateStepsIterator.get ();
    var allDone = true;

    for (var i in upStep.updates) {
      var up = upStep.updates[i];
      var b = balls[up.id];

      if (up.time > b.time) {
        allDone = false; // some ball hasn't progressed to update time

        if (up.delta == 0.0) {  // first time
          assert (up.time > b.time);
          up.delta = (up.time - b.time) / NDELTA;
          b.brighten ();
        }

        b.advanceBy (up.delta);
      } else {
        b.applyUpdate (up);
        b.restoreColor ();
      }
    }

    if (allDone) {
      updateStepsIterator.increment ();
    }
  } else {
    reinitScene();
  }

  renderer.render (scene, camera);
}

// function animateDummy () {
//   requestAnimationFrame (animateDummy);
//   renderer.render (scene, camera);
// }

// init ();
// animateDummy ();
// animate ();
// animateEvents ();

// initRenderer ();

function billiardsDemo (demoType) {
  switch (demoType) {
    case "SYNCHRONOUS":
      reinitScene = function() { initScene ("config.csv", "balls.csv", "simLogSer.csv"); animate(); }
      reinitScene();
      break;
    case "ASYNC_SERIAL":
      reinitScene = function () { initScene ("config.csv", "balls.csv", "simLogSer.csv"); animateEvents(); }
      reinitScene();
      break;
    case "ASYNC_PARALLEL":
      reinitScene = function() { initScene ("config.csv", "balls.csv", "simLogPar.csv"); animateEvents(); }
      reinitScene();
      break;
    default:
      alert ("unknown demoType: " + demoType);
      break;
  }
}

