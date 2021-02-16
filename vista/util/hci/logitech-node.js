/*
This node runs as an interface to the Logitech G29 steering wheel.
It accepts commands from a defined socket and returns the state of the device
or actuates certain elements (ex. force feedback).

Requires: express, logitech-g29
*/


var eRESET = -1;
var eGET_STEERING = 0;
var eGET_THROTTLE = 1;
var eGET_AUTO = 2;
var eSET_FORCE = 3;
var eSET_AUTO = 4;

port_arg = process.argv[2]
if (port_arg) {
  var port = Number(port_arg);
} else {
  var port = 3000;
}

angle_arg = process.argv[3]
if (angle_arg) {
  var steering_range = 2*Number(angle_arg);
} else {
  var steering_range = 1000;
}

verbose_arg = process.argv[4]
if (verbose_arg) {
  var verbose = verbose_arg == '1';
} else {
  var verbose = false;
}


var app = require("express")();
var http = require('http').Server(app);
var bodyParser = require('body-parser');
var g = require('logitech-g29')

var options = {
  autocenter: false,
  debug: false,
  range: steering_range
}

var alpha = 0.97;
var latest_steering = 0;
var latest_throttle = 0;
var latest_auto = 0;

g.connect(options, function(err) {
  g.forceConstant(0.5); // direction
  g.forceFriction(0.0)   // no friction

  g.on('wheel-turn', function(val) {
    latest_steering = alpha*latest_steering + (1-alpha)*val;
  })

  g.on('pedals-gas', function(val) {
    latest_throttle = val;
  })

  g.on('wheel-button_spinner', function(val) {
    if (val==1) {
      latest_auto = (latest_auto + 1) % 2;
      if (latest_auto == 1) {
        g.forceFriction(0.5)   // some dampening
      } else {
        g.forceConstant(0.5); // direction
        g.forceFriction(0.0)   // no friction
      }
    }
  })

  app.use(bodyParser.json())
  app.post('/',function(req,res){
    var cmd = req.body.cmd;

    if (cmd == eRESET) {
      console.log("reset: ");
      latest_auto = 0;
      res.send(1);

    } else if (cmd == eGET_STEERING) {
      angle = (latest_steering - 50.0)/50.0 * steering_range/2.
      // if (verbose) console.log("get steering: ", angle);
      res.send(angle.toString());

    } else if (cmd == eGET_THROTTLE) {
      // if (verbose) console.log("get throttle: ", latest_throttle);
      res.send(latest_throttle.toString());

    } else if (cmd == eGET_AUTO) {
      if (verbose) console.log("get auto: ", latest_auto);
      res.send(latest_auto.toString());

    } else if (cmd == eSET_FORCE) {
      var force = req.body.msg;
      // if (verbose) console.log("set force: ", force);
      g.forceConstant(force); // direction
      res.send(latest_steering.toString());

    } else if (cmd == eSET_AUTO) {
      var auto = req.body.msg;
      if (verbose) console.log("set auto: ", auto);
      latest_auto = Number(auto)
      if (latest_auto == 1) {
        g.forceFriction(0.5)   // some dampening
      } else {
        g.forceConstant(0.5); // direction
        g.forceFriction(0.0)   // no friction
      }
      res.send(auto.toString());
    }

  });

  http.listen(port, function(){
  if (verbose) console.log('listening...');
  });

}) //logitech-g29.connect
