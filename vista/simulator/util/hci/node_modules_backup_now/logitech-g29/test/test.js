//----------
// Includes
//----------
const color = require('../code/color.js')
const g     = require('../code/index.js')

//--------------
// Instructions
//--------------
console.log()
console.log(color.cyan('Play with your G29 to make sure each feature works perfectly.'))
console.log()
console.log(color.cyan('Setting up wheel.'))
console.log(color.cyan('One moment please.'))
console.log(color.gray('If nothing happens, try moving the wheel.'))

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
const options = {
    debug: true
}

//---------
// Connect
//---------
g.connect(options, function(err) {
    if (err) {
        console.log(color.yellow('Oops -> ') + err)
        console.log()
        console.log(color.cyan('The wheel may be busy. Try again in a few seconds.'))
        console.log()
    }
})