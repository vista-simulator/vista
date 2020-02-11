//----------
// Includes
//----------
const hid = require('node-hid')

//----------------------
// Get and Show Devices
//----------------------
const devices = hid.devices()

console.log(devices)