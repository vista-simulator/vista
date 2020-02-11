# API - Logitech G29 Racing Wheel for Node

* [connect](#connect)
  * [options](#options)
* [events](#events)
  * [event map](#event-map)
  * [on](#on)
  * [once](#once)
* [force](#force)
  * [forceConstant](#forceconstant)
  * [forceFriction](#forcefriction)
  * [forceOff](#forceoff)
* [leds](#leds)
* [disconnect](#disconnect)
* [advanced](#advanced)
  * [emitter](#emitter)
  * [relay](#relay)
  * [relayOS](#relayos)

## connect

`connect(callback)` or `connect(options, callback)`

Connect to the wheel and receive a callback once it is ready.

```js
g.connect(function(err) {
  console.log('Ready')
})
```

Connect with `options`.

```js
const options = {
  autocenter: false,
  range: 270
}

g.connect(options, function(err) {
  console.log('Ready')
})
```

### options

The following options can be set when using `connect(options, callback)`.

|Option|Default|Type|Examples|
|:--|:--|:--|:--|
|autocenter|true|boolean or array|true, false, [0.3, 0.7]|
|debug|false|boolean|true, false|
|range|900|number|270, 900|

`autocenter` can be fine tuned if you provide a two element array. The first value (0 - 1) controls the general strength of the auto-centering. The second value (0 - 1) controls how quickly that strength ramps up as you turn the wheel more.

`debug` enables a lot of console logging.

`range` is a number from 270 to 900. Range sets the degrees of turn it takes before the wheel reports a maximum value for that direction. For example, if the range is 270, it won't take much turning before you receive a min or max return value. Even if you can physically turn the wheel more in the same direction, the return value will be the same.

## events

Events can be subscribed to using `on` and `once`.

For example, if you wanted to listen for wheel turns and gear changes you could write:

```js
g.on('wheel-turn', function(val) {
  console.log('Wheel turned to ' + val)
}).on('shifter-gear', function(val) {
  console.log('Shifted into ' + val)
})
```

**Wheel Events**

|Event|Returns|Values|Notes|
|:--|:--|:--|:--|
|`wheel-turn`|number|0 - 100|0 is full right<br>50 is centered<br>100 is full left|
|`wheel-shift_left`|binary|0, 1||
|`wheel-shift_right`|binary|0, 1||
|`wheel-dpad`|number|0 - 8|0 = neutral<br>1 = north<br>2 = northeast<br>3 = east<br>4 = southeast<br>5 = south<br>6 = southwest<br>7 = west<br>8 = northwest|
|`wheel-button_x`|binary|0, 1||
|`wheel-button_square`|binary|0, 1||
|`wheel-button_triangle`|binary|0, 1||
|`wheel-button_circle`|binary|0, 1||
|`wheel-button_l2`|binary|0, 1||
|`wheel-button_r2`|binary|0, 1||
|`wheel-button_l3`|binary|0, 1||
|`wheel-button_r3`|binary|0, 1||
|`wheel-button_plus`|binary|0, 1||
|`wheel-button_minus`|binary|0, 1||
|`wheel-spinner`|number|-1, 0, 1|-1 = left<br>0 = neutral<br>1 = right|
|`wheel-button_spinner`|binary|0, 1||
|`wheel-button_share`|binary|0, 1||
|`wheel-button_option`|binary|0, 1||
|`wheel-button_playstation`|binary|0, 1||

**Shifter Events**

|Event|Returns|Values|Notes|
|:--|:--|:--|:--|
|`shifter-gear`|number|0 - 6, -1|0 = neutral<br>1-6 = gears<br>-1 = reverse|

**Pedal Events**

|Event|Returns|Values|Notes|
|:--|:--|:--|:--|
|`pedals-gas`|number|0 - 1|0 is no pressure, 0.25 is quarter pressure, and 1 is fully pressed.|
|`pedals-brake`|number|0 - 1|0 is no pressure, 0.25 is quarter pressure, and 1 is fully pressed.|
|`pedals-clutch`|number|0 - 1|0 is no pressure, 0.25 is quarter pressure, and 1 is fully pressed.|

Not enough events for you? Try subscribing to `all`, `changes`, or `data` for even more information.

### event map

[![Event Map](https://raw.github.com/nightmode/logitech-g29/master/images/event-map.png)](https://raw.github.com/nightmode/logitech-g29/master/images/event-map.png)

### on

`on(event, callback)`

Can be specified before or after a `connect(callback)`.

```js
g.on('wheel-button_playstation', function(val) {
  if (val) {
      console.log('I really love it when you press my buttons.')
  }
})
```


### once

`once(event, callback)`

Can be specified before or after a `connect(callback)`.

```js
g.once('pedals-gas', function(val) {
    // the following message will only be displayed once
    console.log('Powered by dead dinosaur juice, your engine roars to life!')
})
```

## force

### forceConstant

`forceConstant(number)` where number is 0 - 1 to indicate both direction and strength.

```js
g.forceConstant()    // no force
g.forceConstant(0)   // full left
g.forceConstant(0.5) // no force
g.forceConstant(1)   // full right
```

### forceFriction

`forceFriction(number)` where number is 0 - 1 to indicate effect strength.

```js
g.forceFriction()    // no friction
g.forceFriction(0)   // no friction
g.forceFriction(0.5) // half strength
g.forceFriction(1)   // full strength
```

### forceOff

`forceOff()`

Turn off all force effects except auto-centering.

### leds

`leds()` or `leds(number)` or `leds(string)` or `leds(array)`

The shift indicator LEDs can be interfaced with in a variety of ways.

`led()` is the easiest way to turn off all LEDs.

`led(number)` where number is between 0 - 1 to indicate a percent.

```js
g.led(0.45) // the least accurate way to control LEDs since an arbitrary scale will be used for conversion
```

`led(string)` where string is zero to five characters of zeroes or ones.

```js
g.leds('')      // all off
g.leds('1')     // green
g.leds('111')   // green, green, orange
g.leds('00001') // red only
```

`led(array)` where array is zero to five elements of zeroes or ones.

```js
g.leds([])          // all off
g.leds([1])         // green
g.leds([1,1,1])     // green, green, orange
g.leds([0,0,0,0,1]) // red only
```

## disconnect

`disconnect()`

Disconnect in preparation to connect again or to allow other software to use the wheel.

## Advanced

### emitter

`emitter.` + [nodejs.org/api/events.html](https://nodejs.org/api/events.html)

Exposes the EventEmitter that this library uses behind the scenes.

```js
g.emitter.removeAllListeners()
```

### relay

`relay(data)`

Relay low level commands directly to the hardware.

```js
// turn on all LEDs
g.relay([0x00, 0xf8, 0x12, 0x1f, 0x00, 0x00, 0x00, 0x01])
```

### relayOS

`relayOS(data)`

Relay low level commands directly to the hardware after applying OS specific tweaks, if needed.

```js
// turn on all LEDs
g.relayOS([0xf8, 0x12, 0x1f, 0x00, 0x00, 0x00, 0x01])
```

## License

MIT © [Kai Nightmode](https://twitter.com/kai_nightmode)