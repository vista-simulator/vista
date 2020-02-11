//----------
// Includes
//----------
var chalk = require('chalk')
var g = require('../code/index.js')

//-----------
// Variables
//-----------
displayNone  = chalk.black('*****')
displayOne   = chalk.green('*') + chalk.black('****')
displayTwo   = chalk.green('**') + chalk.black('***')
displayThree = chalk.green('**') + chalk.yellow('*') + chalk.black('**')
displayFour  = chalk.green('**') + chalk.yellow('**') + chalk.black('*')
displayFive  = chalk.green('**') + chalk.yellow('**') + chalk.red('*')

//-----------
// Functions
//-----------
function display(val) {
    val  = Math.round(val * 100)

    if (val > 84) {
        console.log(displayFive)
    } else if (val > 69) {
        console.log(displayFour)
    } else if (val > 39) {
        console.log(displayThree)
    } else if (val > 19) {
        console.log(displayTwo)
    } else if (val > 4) {
        console.log(displayOne)
    } else {
        console.log(displayNone)
    }
}

//----------------------------------
// Distract the humans for a moment
//----------------------------------
console.log(chalk.cyan('Setting up wheel.'))
console.log(chalk.cyan('One moment please.'))
console.log(chalk.gray('If nothing happens, try moving the wheel.'))

//-----------------------------------
// Connect to Wheel and Setup Events
//-----------------------------------
g.connect(function(err) {
    g.on('pedals-gas', function(val) {
        display(val)
        g.leds(val)
    })

    console.log(chalk.cyan('Wheel ready.'))
    console.log(chalk.green('Press the gas pedal to see if your wheel LEDs animate.'))
})