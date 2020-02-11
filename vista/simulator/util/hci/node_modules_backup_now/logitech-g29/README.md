<img src="https://raw.githubusercontent.com/nightmode/logitech-g29/master/images/header.png" width="888" alt="">

# Logitech G29 Racing Wheel for Node

Bring your [Logitech G29 Racing Wheel](https://www.logitechg.com/en-us/products/driving/driving-force-racing-wheel.html) into the wonderful world of [Node](https://nodejs.org/en/).

* Subscribe to wheel, pedal, and shifter events.
* Activate simple force feedback effects.
* Set wheel auto-centering and range.
* Customize shift indicator LEDs.

## Requirements

[Node](https://nodejs.org/en/) version 8 or greater.

Make sure your wheel's [platform mode switch](http://support.logitech.com/en_us/article/Set-the-G29-racing-wheel-for-Playstation-4-Playstation-3-or-PC-platforms?product=a0qi0000006PmxKAAS) is set to PS3.

## Install

This library uses [node-hid](https://github.com/node-hid/node-hid) behind the scenes. Depending on your OS and Node version, you may have an effortless install. If not, you may want to consult node-hid's [compiling from source](https://github.com/node-hid/node-hid#compiling-from-source) guide for assistance.

```
npm install logitech-g29
```

[Ubuntu](http://www.ubuntu.com/desktop) users will most likely want to remove the `sudo` requirement of interfacing with the wheel. This can be easily accomplished by creating a file at `/etc/udev/rules.d/99-hidraw-permissions.rules` with the following code. After saving the file, reboot and then you can move on to more fun tasks.

```
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", MODE="0664", GROUP="plugdev"
```

## Example

Let's have some fun and make our wheel LEDs light up when we press the gas pedal.

```js
const g = require('logitech-g29')

g.connect(function(err) {
    g.on('pedals-gas', function(val) {
        g.leds(val)
    })
})
```

Vroom vroom sounds optional but encouraged. ^\_^

## API

* [connect](docs/api.md#connect)
  * [options](docs/api.md#options)
* [events](docs/api.md#events)
  * [event map](docs/api.md#event-map)
  * [on](docs/api.md#on)
  * [once](docs/api.md#once)
* [force](docs/api.md#force)
  * [forceConstant](docs/api.md#forceconstant)
  * [forceFriction](docs/api.md#forcefriction)
  * [forceOff](docs/api.md#forceoff)
* [leds](docs/api.md#leds)
* [disconnect](docs/api.md#disconnect)
* [advanced](docs/api.md#advanced)
  * [emitter](docs/api.md#emitter)
  * [relay](docs/api.md#relay)
  * [relayOS](docs/api.md#relayos)

## Support

Help [support this project](https://www.patreon.com/nightmode) on Patreon. Help me caretake and craft new software, videos, and interactive art. All for as little as $1 a month.

In addition to Patreon, here are some other ways you can help this project.

* [Report any issues](https://github.com/nightmode/logitech-g29/issues) on GitHub.
* Share your G29 wheel creation on social media to inspire others.

## License

MIT © [Kai Nightmode](https://twitter.com/kai_nightmode)