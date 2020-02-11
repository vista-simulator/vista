'use strict'

//----------
// Data Map
//----------
/*
Details on each item of the read buffer provided by node-hid for the Logitech G29.

    Zero
        Wheel - Dpad
            0 = Top
            1 = Top Right
            2 = Right
            3 = Bottom Right
            4 = Bottom
            5 = Bottom Left
            6 = Left
            7 = Top Left
            8 = Dpad in Neutral Position

        Wheel - Symbol Buttons
           16 = X
           32 = Square
           64 = Circle
          128 = Triangle

    One
        Wheel - Shifter Pedals
            1 = Right Shifter
            2 = Left Shifter

        Wheel - Buttons
            4 = R2 Button
            8 = L2 Button
           16 = Share Button
           32 = Option Button
           64 = R3 Button
          128 = L3 Button

    Two
        Shifter - Gear Selector
             0 = Neutral
             1 = 1st Gear
             2 = 2nd Gear
             4 = 3rd Gear
             8 = 4th Gear
            16 = 5th Gear
            32 = 6th Gear
            64 = Reverse Gear

        Wheel
           128 = Plus Button

    Three
        Wheel - Spinner and Buttons
            1 = Minus Button
            2 = Spinner Right
            4 = Spinner Left
            8 = Spinner Button
           16 = PlayStation Button

    Four
        Wheel - Wheel Turn (fine movement)
            0-255

            0 is far left
            255 is far right

    Five
        Wheel - Wheel Turn
            0-255

            0 is far left
            255 is far right

    Six
        Pedals - Gas
            0-255

            0 is full gas
            255 is no pressure

    Seven
        Pedals - Brake
            0-255

            0 is full brake
            255 is no pressure

    Eight
        Pedals - Clutch
            0-255

            0 is full clutch
            255 is no pressure

    Nine
        Shifter
            X Coordinates (not used)

    Ten
        Shifter
            Y Coordinates (not used)

    Eleven
        Shifter
            Contains data on whether or not the gear selector is pressed down into the unit.
            If pressed down, the user is probably preparing to go into reverse. (not used)
*/

//-----------
// Functions
//-----------
function dataMap(dataDiffPositions, data, memory) {
    /*
    Figure out what has changed since the last event and call relevent functions to translate those changes to a memory object.
    @param   {Object}  dataDiffPositions  An array.
    @param   {Buffer}  data               Buffer data from a node-hid event.
    @param   {Object}  memory             Memory object to modify.
    @return  {Object}  memory             Modified memory object.
    */
    for (var i in dataDiffPositions) {
        switch (dataDiffPositions[i]) {
            case 0:
                memory = wheelDpad(data, memory)
                memory = wheelButtonsSymbols(data, memory)
                break
            case 1:
                memory = wheelShiftPedals(data, memory)
                memory = wheelButtons(data, memory)
                break
            case 2:
                memory = shifterGear(data, memory)
                memory = wheelButtonPlus(data, memory)
                break
            case 3:
                memory = wheelSpinnerAndButtons(data, memory)
                break
            case 4:
            case 5:
                memory = wheelTurn(data, memory)
                break
            case 6:
                memory = pedalsGas(data, memory)
                break
            case 7:
                memory = pedalsBrake(data, memory)
                break
            case 8:
                memory = pedalsClutch(data, memory)
                break
            case 11:
                memory = shifterGear(data, memory) // for reverse
        }
    }

    return memory
} // dataMap

function reduceNumberFromTo(num, to) {
    /*
    Reduce a number by 128, 64, 32, etc... without going lower than a second number.
    @param   {Number}  num
    @param   {Number}  to
    @return  {Number}
    */
    to = to * 2

    var y = 128

    while (y > 1) {
        if (num < to) {
            break
        }

        if (num - y >= 0) {
            num = num - y
        }

        y = y / 2
    }

    return num
} // reduceNumberFromTo

function round(num, exp) {
    /*
    Round a number to a certain amount of places.
    @param   {Number}  num  Number like 1.567.
    @param   {Number}  exp  Number of places to round to.
    @return  {Number}
    */
    if (typeof exp === 'undefined' || +exp === 0) {
        return Math.round(num)
    }

    num = +num
    exp = +exp

    if (isNaN(num) || !(typeof exp === 'number' && exp % 1 === 0)) {
        return NaN
    }

    // Shift
    num = num.toString().split('e')
    num = Math.round(+(num[0] + 'e' + (num[1] ? (+num[1] + exp) : exp)))

    // Shift back
    num = num.toString().split('e')
    return +(num[0] + 'e' + (num[1] ? (+num[1] - exp) : -exp))
} // round

//------------------
// Functions: Wheel
//------------------
function wheelButtonPlus(data, memory) {
    var d = data[2]

    memory.wheel.button_plus = (d & 128) ? 1 : 0

    return memory
} // wheelButtonPlus

function wheelButtons(data, memory) {
    var d = data[1]

    memory.wheel.button_r2 = (d & 4) ? 1 : 0
    memory.wheel.button_l2 = (d & 8) ? 1 : 0

    memory.wheel.button_share = (d & 16) ? 1 : 0
    memory.wheel.button_option = (d & 32) ? 1 : 0
    memory.wheel.button_r3 = (d & 64) ? 1 : 0
    memory.wheel.button_l3 = (d & 128) ? 1 : 0

    return memory
} // wheelButtons

function wheelButtonsSymbols(data, memory) {
    var d = data[0]

    memory.wheel.button_x = (d & 16) ? 1 : 0
    memory.wheel.button_square = (d & 32) ? 1 : 0
    memory.wheel.button_circle = (d & 64) ? 1 : 0
    memory.wheel.button_triangle = (d & 128) ? 1 : 0

    return memory
} // wheelButtonsSymbols

function wheelDpad(data, memory) {
    var dpad = reduceNumberFromTo(data[0], 8)

    switch (dpad) {
        case 8:
            // neutral
            memory.wheel.dpad = 0
            break
        case 7:
            // top left
            memory.wheel.dpad = 8
            break
        case 6:
            // left
            memory.wheel.dpad = 7
            break
        case 5:
            // bottom left
            memory.wheel.dpad = 6
            break
        case 4:
            // bottom
            memory.wheel.dpad = 5
            break
        case 3:
            // bottom right
            memory.wheel.dpad = 4
            break
        case 2:
            // right
            memory.wheel.dpad = 3
            break
        case 1:
            // top right
            memory.wheel.dpad = 2
            break
        case 0:
            // top
            memory.wheel.dpad = 1
    }

    return memory
} // wheelDpad

function wheelShiftPedals(data, memory) {
    var d = data[1]

    memory.wheel.shift_right = d & 1
    memory.wheel.shift_left = (d & 2) ? 1 : 0

    return memory
} // wheelShiftPedals

function wheelSpinnerAndButtons(data, memory) {
    var d = data[3]

    memory.wheel.button_minus = d & 1

    if (d & 2) {
        memory.wheel.spinner = 1
    } else if (d & 4) {
        memory.wheel.spinner = -1
    } else {
        memory.wheel.spinner = 0
    }

    memory.wheel.button_spinner = (d & 8) ? 1 : 0
    memory.wheel.button_playstation = (d & 16) ? 1 : 0

    return memory
} // wheelSpinnerAndButtons

function wheelTurn(data, memory) {
    var wheelCourse = data[5] // 0-255
    var wheelFine = data[4] // 0-255

    wheelCourse = wheelCourse / 255 * 99 // 99 instead of 100 so wheelCourse and wheelFine add up to 100% when they are both maxed out
    wheelFine = wheelFine / 255

    var wheel = round(wheelCourse + wheelFine, 2)

    if (wheel > 100) wheel = 100

    if (wheel < 0) wheel = 0

    memory.wheel.turn = wheel

    return memory
} // wheelTurn

//-------------------
// Functions: Pedals
//-------------------
function pedalsBrake(data, memory) {
    memory.pedals.brake = pedalToPercent(data[7])
    return memory
} // pedalsBrake


function pedalsClutch(data, memory) {
    memory.pedals.clutch = pedalToPercent(data[8])
    return memory
} // pedalsClutch

function pedalsGas(data, memory) {
    memory.pedals.gas = pedalToPercent(data[6])
    return memory
} // pedalsGas

function pedalToPercent(num) {
    // invert numbers
    num = Math.abs(num - 255)

    // change to a percent like 0 for no pressure, 0.5 for half pressure, and 1 for full pressure
    num = round(num / 255, 2)

    return num
} // pedalToPercent

//--------------------
// Functions: Shifter
//--------------------
function shifterGear(data, memory) {
    var stick = data[2]

    stick = reduceNumberFromTo(stick, 64)

    switch (stick) {
        case 0:
            // neutral
            memory.shifter.gear = 0
            break
        case 1:
            // first gear
            memory.shifter.gear = 1
            break
        case 2:
            // second gear
            memory.shifter.gear = 2
            break
        case 4:
            // third gear
            memory.shifter.gear = 3
            break
        case 8:
            // fourth gear
            memory.shifter.gear = 4
            break
        case 16:
            // fifth gear
            memory.shifter.gear = 5
            break
        case 32:
            // sixth gear
            memory.shifter.gear = 6
            break
        case 64:
            // reverse gear
            memory.shifter.gear = -1
    }

    return memory
} // shifterGear

//---------
// Exports
//---------
module.exports = dataMap