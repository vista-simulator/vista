//----------
// Includes
//----------
var hid = require('node-hid')

//----------------------
// Get and Show Devices
//----------------------
var devices = hid.devices()

console.log(devices)