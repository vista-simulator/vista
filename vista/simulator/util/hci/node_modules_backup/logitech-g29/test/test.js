//----------
// Includes
//----------
var chalk = require('chalk')

//--------------
// Instructions
//--------------
console.log()
console.log(chalk.cyan('Play with your G29 to make sure each feature works perfectly.'))
console.log()
console.log(chalk.cyan('Setting up wheel.'))
console.log(chalk.cyan('One moment please.'))
console.log(chalk.gray('If nothing happens, try moving the wheel.'))

//----------------
// Includes: Self
//----------------
var g = require('./../code/index.js')

//------------------
// Graceful Exiting
//------------------
process.on('SIGINT', function() {
    g.disconnect()
    process.exit()
})

//-----------
// Variables
//-----------
var options = {
    debug: true
}

//---------
// Connect
//---------
g.connect(options, function(err) {
    if (err) {
        console.log(chalk.yellow('Oops -> ') + err)
        console.log()
        console.log(chalk.cyan('The wheel may be busy. Try again in a few seconds.'))
        console.log()
    }
})