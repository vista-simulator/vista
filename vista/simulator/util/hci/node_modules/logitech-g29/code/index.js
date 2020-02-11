'use strict'

//----------------
// Includes: Self
//----------------
const color   = require('./color.js')
const dataMap = require('./data-map.js')

//----------
// Includes
//----------
const events = require('events')
const hid = require('node-hid')
const os = require('os')

//-----------
// Variables
//-----------
const eventEmitter = new events.EventEmitter()
const options = {
    'autocenter': true,
    'debug': false,
    'range': 900
}
const platform = os.platform()

let dataPrev = Array(12)
let device = ''
let ledPrev = []
let memoryPrev = {
    'wheel': {
        'turn': 50,
        'shift_left': 0,
        'shift_right': 0,
        'dpad': 0,
        'button_x': 0,
        'button_square': 0,
        'button_triangle': 0,
        'button_circle': 0,
        'button_l2': 0,
        'button_r2': 0,
        'button_l3': 0,
        'button_r3': 0,
        'button_plus': 0,
        'button_minus': 0,
        'spinner': 0,
        'button_spinner': 0,
        'button_share': 0,
        'button_option': 0,
        'button_playstation': 0
    },
    'shifter': {
        'gear': 0
    },
    'pedals': {
        'gas'   : 0,
        'brake' : 0,
        'clutch': 0
    }
}
let prependWrite = false

//-------------
// OS Specific
//-------------
if (platform === 'win32' || platform === 'win64') {
    prependWrite = true
}

//-----------
// Functions
//-----------
function clone(obj) {
    /*
    Clone an object.
    @param   {Object}  obj  Object to clone.
    @return  {Object}
    */
    if (obj === null || typeof obj !== 'object') {
        return obj
    }

    const temp = obj.constructor()

    for (let key in obj) {
        temp[key] = clone(obj[key])
    }

    return temp
} // clone

function connect(odo, callback) { // Constable Odo takes many forms.
    /*
    Connect to a Logitech G29 wheel.
    @param   {Object, Function}  odo       Options object or callback function.
    @param   {Function}          callback  Callback function.
    */
    if (typeof odo === 'function') {
        callback = odo
    } else {
        userOptions(odo)
    }

    callback = (typeof callback === 'function') ? callback : function() {}

    device = new hid.HID(findWheel())

    device.read(function(err, data) {
        if (err) {
            if (options.debug) {
                console.log(color.red('connect -> Error reading from device.'), err)
            }
            callback(err)
        } else {
            forceOff()

            if (data.length === 12) {
                // wheel is already in high precision mode

                if (options.debug) {
                    console.log(color.cyan('connect -> Wheel already in high precision mode.'))
                }

                listen(true, callback)
            } else {
                // wheel is not in high precision mode

                if (options.debug) {
                    console.log(color.cyan('connect -> Initing'))
                }

                try {
                    // G29 Wheel init from - https://github.com/torvalds/linux/blob/master/drivers/hid/hid-lg4ff.c
                    relayOS([0xf8, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00])
                    relayOS([0xf8, 0x09, 0x05, 0x01, 0x01, 0x00, 0x00])

                    // wait for wheel to finish calibrating
                    setTimeout(function() {
                        listen(false, callback)
                    }, 8000)
                } catch(err) {
                    callback(err)
                }
            }
        }
    })

    forceConstant(1) // move the wheel to generate a read event
} // connect

function disconnect() {
    /*
    Disconnect in preparation to connect again or to allow other software to use the wheel.
    */
    device.close()
} // disconnect

function findWheel() {
    /*
    Return the USB location of a Logitech G29 wheel.
    @return  {String}  devicePath  USB path like: USB_046d_c294_fa120000
    */
    const devices = hid.devices()
    let devicePath = ''

    for (let i in devices) {
        // devices[i].vendorId seems to be the only completely reliable property on each OS.
        // devices[i].productId can not be trusted and can sometimes be wildly different.
        // devices[i].product should be set to 'G29 Driving Force Racing Wheel'.
        // devices[i].interface should be 0 on Windows and Linux.
        // devices[i].usagePage should be 1 on Windows and Mac.
        if (devices[i].vendorId === 1133 &&
            (devices[i].productId === 49743 || devices[i].product === 'G29 Driving Force Racing Wheel') &&
            (devices[i].interface === 0 || devices[i].usagePage === 1)) {
            devicePath = devices[i].path
            break
        }
    }

    if (devicePath === '') {
        if (options.debug) {
            console.log(color.yellow('findWheel -> Oops, could not find a G29 Wheel. Is it plugged in?\n'))
            process.exit()
        }
    } else if (options.debug) {
        console.log(color.cyan('findWheel -> Found G29 Wheel at ') + devicePath)
    }

    return devicePath
} // findWheel

function on(str, func) {
    return eventEmitter.on(str, func)
} // on

function once(str, func) {
    return eventEmitter.once(str, func)
} // once

function relay(data) {
    /*
    Relay low level commands directly to the hardware.
    @param  {Object}  data  Array of data to write. For example: [0x00, 0xf8, 0x12, 0x1f, 0x00, 0x00, 0x00, 0x01]
    */
    if (Array.isArray(data)) {
        device.write(data)
    }
} // relay

function relayOS(data) {
    /*
    Relay low level commands directly to the hardware after applying OS specific tweaks, if needed.
    @param  {Object}  data  Array of data to write. For example: [0xf8, 0x12, 0x1f, 0x00, 0x00, 0x00, 0x01]
    */
    if (prependWrite) {
        data.unshift(0x00)
    }

    device.write(data)
}

function setRange() {
    /*
    Set wheel range.
    */
    if (options.range < 270) {
        options.range = 270
    }

    if (options.range > 900) {
        options.range = 900
    }

    const range1 = options.range & 0x00ff
    const range2 = (options.range & 0xff00) >> 8

    relayOS([0xf8, 0x81, range1, range2, 0x00, 0x00, 0x00])
} // setRange

function userOptions(opt) {
    /*
    Set user options.
    @param  {Object}  opt   Options object originally passed into the connect function.
    */
    if (typeof opt !== 'object') return;

    for (let i in options) {
        if (opt.hasOwnProperty(i)) {
            options[i] = opt[i]
        }
    }

    if (options.debug) {
        console.log(color.cyan('userOptions -> '), options)
    }
} // userOptions

//----------------
// Function: LEDs
//----------------
function leds(setting) {
    /*
    Control the shift indicator LEDs using a variety of convience methods.
    @param  {*}  setting  String, Number, or Array setting. Optional. See API documentation for more info.
    */

    // no setting
    if (typeof setting === 'undefined') {
        setting = []
    }

    // percent based settings
    if (typeof setting === 'number') {
        setting = Math.round(setting * 100)

        if (setting > 84) {
            setting = '11111'
        } else if (setting > 69) {
            setting = '1111'
        } else if (setting > 39) {
            setting = '111'
        } else if (setting > 19) {
            setting = '11'
        } else if (setting > 4) {
            setting = '1'
        } else {
            setting = ''
        }
    }

    // string based settings
    if (typeof setting === 'string') {
        setting = setting.split('')
    }

    // array based settings
    if (Array.isArray(setting)) {
        if (ledPrev === setting) {
            return
        }

        const ledValues = [1, 2, 4, 8, 16]

        const ledArray = setting

        // remove any extra elements
        ledArray.splice(5, ledArray.length - 5)

        const len = ledArray.length

        setting = 0

        for (let i = 0; i < len; i++) {
            if (parseInt(ledArray[i]) === 1) {
                setting = setting + ledValues[i]
            }
        }

        /*
        Setting should be a number from 0 to 31

            From outside in, mirrored on each side.

            0 = No LEDs
            1 = Green One
            2 = Green Two
            4 = Orange One
            8 = Orange Two
            16 = Red

            31 = All LEDs
        */

        try {
            relayOS([0xf8, 0x12, setting, 0x00, 0x00, 0x00, 0x01])

            // update global variable for next time
            ledPrev = setting
        } catch(err) {
            // do nothing
        }
    }
} // leds

//------------------
// Functions: Force
//------------------
function autoCenter() {
    /*
    Set wheel autocentering based on existing options.
    */
    const option = options.autocenter

    if (option) {
        // auto-center on
        relayOS([0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

        if (Array.isArray(option) && option.length === 2) {
            // custom auto-center

            // byte 3-4 is effect strength, 0x00 to 0x0f
            option[0] = Math.round(option[0] * 15)

            // byte 5 is the rate the effect strength rises as the wheel turns, 0x00 to 0xff
            option[1] = Math.round(option[1] * 255)

            relayOS([0xfe, 0x0d, option[0], option[0], option[1], 0x00, 0x00, 0x00])
        } else {
            // use default strength profile
            relayOS([0xfe, 0x0d, 0x07, 0x07, 0xff, 0x00, 0x00, 0x00])
        }
    } else {
        // auto-center off
        relayOS([0xf5, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    }
} // autoCenter

function forceConstant(number) {
    /*
    Set or disable a constant force effect.
    @param  {Number}  number  Number between 0 and 1. Optional.
    */
    if (typeof number === 'undefined') number = 0.5

    if (number === 0.5) {
        forceOff(1)
        return
    }

    number = Math.round(Math.abs(number - 1) * 255)

    relayOS([0x11, 0x00, number, 0x00, 0x00, 0x00, 0x00])
} // forceConstant

function forceFriction(number) {
    /*
    Set or disable the ammount of friction present when turning the wheel.
    @param  {Number}  number  Number between 0 and 1. Optional.
    */
    if (typeof number === 'undefined') number = 0

    if (number === 0) {
        forceOff(2)
        return
    }

    // sending manual relay() commands to the hardware seems to reveal a 0x00 through 0x07 range
    // 0x07 is the strongest friction and then 0x08 is no friction
    // friction ramps up again from 0x08 to 0x0F
    number = Math.round(number * 7)

    // the first "number" is for left rotation, the second for right rotation
    relayOS([0x21, 0x02, number, 0x00, number, 0x00, 0x00])
} // forceFriction

function forceOff(slot) {
    /*
    Turn off all force effects except auto-centering.
    @param  {Number}  slot  Number between 0 and 4. Optional.
    */
    // Great info at http://wiibrew.org/wiki/Logitech_USB_steering_wheel, especially about writing to more than one effect slot.
    if (typeof slot === 'undefined') {
        slot = 0xf3
    } else {
        if (slot === 0) {
            slot = 0xf3
        } else {
            slot = parseInt('0x' + slot + '0')
        }
    }

    // turn off effects (except for auto-center)
    relayOS([slot, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
} // forceOff

//------------------
// Function: Listen
//------------------
function listen(ready, callback) {
    /*
    @param  {Boolean}   ready     True if the wheel is ready for commands. Optional.
    @param  {Function}  callback  Optional callback function.
    */
    if (!ready) {
        device.close()
        device = new hid.HID(findWheel())
    }

    setRange()
    autoCenter()

    device.on("data", function(data) {
        // reset memory
        let memory = clone(memoryPrev)
        const memoryCache = clone(memoryPrev)

        const dataDiffPositions = []

        // find out if anything has changed since the last event
        const dataLength = data.length
        for (let i = 0; i < dataLength; i++) {
            if (data[i] !== dataPrev[i]) {
                dataDiffPositions.push(i)
            }
        }

        if (dataDiffPositions.length === 0) {
            return
        }

        memory = dataMap(dataDiffPositions, data, memory)

        //-------------------------
        // Figure out what changed
        //-------------------------
        const memoryDiff = {}
        let count = 0

        for (let o in memoryCache) {
            for (let y in memory[o]) {
                if (memory[o][y] != memoryCache[o][y]) {
                    if (!memoryDiff.hasOwnProperty(o)) {
                        memoryDiff[o] = {}
                    }
                    eventEmitter.emit(o + '-' + y, memory[o][y]) // for example, wheel-turn
                    memoryDiff[o][y] = memory[o][y]
                    count = count + 1
                }
            }
        }

        if (count > 0) {
            if (options.debug) {
                console.log(memoryDiff)
            }

            // emit changes only
            eventEmitter.emit('changes', memoryDiff)
        }

        // emit everything in all event
        eventEmitter.emit('all', memory)

        // emit raw data
        eventEmitter.emit('data', data)

        // set global variables for next event
        memoryPrev = memory
        dataPrev = data
    })

    device.on("error", function(err) {
        if (options.debug) {
            console.log(color.red('device error -> '), JSON.stringify(err), err)
        }
    })

    leds(0)

    if (options.debug) {
        console.log(color.cyan('listen -> listening'))
    }

    callback(null)
} // listen

//---------
// Exports
//---------
module.exports.connect = connect
module.exports.disconnect = disconnect

// events
module.exports.emitter = eventEmitter
module.exports.on = on
module.exports.once = once

// leds
module.exports.leds = leds

// force
module.exports.forceConstant = forceConstant
module.exports.forceFriction = forceFriction
module.exports.forceOff = forceOff

// advanced
module.exports.relay = relay
module.exports.relayOS = relayOS