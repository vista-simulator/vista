//----------
// Includes
//----------
var chalk = require('chalk')
var g = require('../code/index.js')

//-----------
// Variables
//-----------
var defaults = {
    forceConstant: 0.5, // 0 = full left, 0.5 = no force, 1 = full right
    forceFriction: 0    // 0 = no friction, 0.5 = half strength, 1 = full strength
}

var lastDirection = defaults.forceConstant
var lastFriction = defaults.forceFriction

var pedalStatus = {
    brake: false, // boolean
    clutch: 0,    // 0-100
    gas: 0        // 0-100
}

var wheelOptions = {
    autocenter: false,
    debug: false,
    range: 900
}

//-----------
// Functions
//-----------
function mathRound(number) {
    return Math.round(number * 100) / 100;
} // mathRound

function setForceConstant() {
    let direction = defaults.forceConstant

    if (pedalStatus.brake === false) {
        direction += pedalStatus.gas    / 2 / 100 // 0-100 / 2 / 100 so 0.5 is the max difference
        direction -= pedalStatus.clutch / 2 / 100 // 0-100 / 2 / 100 so 0.5 is the max difference

        direction = mathRound(direction)

        if (direction < 0) direction = 0
        if (direction > 1) direction = 1
    }

    if (direction !== lastDirection) {
        console.log(chalk.gray('forceConstant(' + direction + ')'))

        g.forceConstant(direction)

        lastDirection = direction
    }
} // setForceConstant

function setForceFriction(val) {
    if (val === 0) {
        lastFriction = 0
    } else if (val > 0) {
        lastFriction += 0.1
    } else {
        lastFriction -= 0.1
    }

    lastFriction = mathRound(lastFriction)

    if (lastFriction < 0) lastFriction = 0
    if (lastFriction > 1) lastFriction = 1

    console.log(chalk.gray('forceFriction(' + lastFriction + ')'))

    g.forceFriction(lastFriction)
} // setForceFriction

//----------------------------------
// Distract the humans for a moment
//----------------------------------
console.log(chalk.cyan('Setting up wheel.'))
console.log(chalk.cyan('One moment please.'))
console.log(chalk.gray('If nothing happens, try moving the wheel.'))

//-----------------------------------
// Connect to Wheel and Setup Events
//-----------------------------------
g.connect(wheelOptions, function(err) {
    g.on('pedals-clutch', function(val) {
        val = val * 100
        if (val >= 10) {
            pedalStatus.clutch = val
        } else {
            pedalStatus.clutch = 0
        }
        setForceConstant()
    })

    g.on('pedals-brake', function(val) {
        val = val * 100
        if (val >= 10) {
            pedalStatus.brake = true
        } else {
            pedalStatus.brake = false
        }
        setForceConstant()
    })

    g.on('pedals-gas', function(val) {
        val = val * 100
        if (val >= 10) {
            pedalStatus.gas = val
        } else {
            pedalStatus.gas = 0
        }
        setForceConstant()
    })

    g.on('wheel-spinner', function(val) {
        if (val !== 0) {
            setForceFriction(val)
        }
    })

    g.on('wheel-button_spinner', function(val) {
        if (val === 1) {
            setForceFriction(0)
        }
    })

    console.log(chalk.cyan('Wheel ready.'))
    console.log()
    console.log(chalk.green('Play with forceConstant() by using the Clutch, Brake, and Gas pedals.'))
    console.log()
    console.log(chalk.green('Play with forceFriction() by using the Red Spinner. Rotate right for more, left for less, and press the spinner button to reset.'))
    console.log()

    // monitor over time
    setInterval(function() {
        setForceConstant()
    }, 300)
})