FluidModule = function(canvas) {

var cat = function() {
    var r = "";
    for (var i = 0; i < arguments.length; i++) {
        r += arguments[i];
    }
    return r;
};

var shader_vs = <<SHADER_VS;
    attribute vec3 aPos;
    attribute vec2 aTexCoord;
    varying   vec2 uv;
    varying vec2 uv_orig;
    void main(void) {
       gl_Position = vec4(aPos, 1.);
       uv = aTexCoord;
       uv_orig = uv;
    }
    SHADER_VS

var shader_fs_inc = <<SHADER_FS_INC;
    #ifdef GL_ES
    precision mediump float;
    #endif

    varying vec2 uv;
    varying vec2 uv_orig;

    vec2 encode(float v){
        vec2 c = vec2(0.);

        int signum = (v >= 0.) ? 128 : 0;
        v = abs(v);
        int exponent = 15;
        float limit = 64.; // considering the bias from 2^-9 to 2^6 (==64)
        for(int exp = 15; exp > 0; exp--){
            if( v < limit){
                limit /= 2.;
                exponent--;
            }
        }

        float rest;
        if(exponent == 0){
            rest = v / limit / 2.;		// "subnormalize" implicite preceding 0. 
        }else{
            rest = (v - limit)/limit;		// normalize accordingly to implicite preceding 1.
        }

        int mantissa = int(rest * 2048.);	// 2048 = 2^11 for the (split) 11 bit mantissa
        int msb = mantissa / 256;			// the most significant 3 bits go into the lower part of the first byte
        int lsb = mantissa - msb * 256;		// there go the other 8 bit of the lower significance

        c.x = float(signum + exponent * 8 + msb) / 255.;	// yeah, the '+1)/255.' seems a little bit odd, but it turned out necessary on my AMD Radeon HD series
        c.y = float(lsb) / 255.;							// ^^ ...same weird color normalization for texture2D here

        if(v >= 2048.){
            //c.x = float( 128. + float(signum)) / 256.;
            c.y = 1.;
        }

        return c;
    }

    float decode(vec2 c){
        float v = 0.;

        int ix = int(c.x*255.); // 1st byte: 1 bit signum, 4 bits exponent, 3 bits mantissa (MSB)
        int iy = int(c.y*255.);	// 2nd byte: 8 bit mantissa (LSB)

        int s = (c.x >= 0.5) ? 1 : -1;
        ix = (s > 0) ? ix - 128 : ix; // remove the signum bit from exponent
        int iexp = ix / 8; // cut off the last 3 bits of the mantissa to select the 4 exponent bits
        int msb = ix - iexp * 8;	// subtract the exponent bits to select the 3 most significant bits of the mantissa

        int norm = (iexp == 0) ? 0 : 2048; // distinguish between normalized and subnormalized numbers
        int mantissa = norm + msb * 256 + iy; // implicite preceding 1 or 0 added here
        norm = (iexp == 0) ? 1 : 0; // normalization toggle
        float exponent = pow( 2., float(iexp + norm) - 20.); // -9 for the the exponent bias from 2^-9 to 2^6 plus another -11 for the normalized 12 bit mantissa 
        v = float( s * mantissa ) * exponent;

        return v;
    }

    vec4 encode2(vec2 v){
        return vec4( encode(v.x), encode(v.y) );
    }
            
    vec2 decode2(vec4 c){
        return vec2( decode(c.rg), decode(c.ba) );
    }

    bool is_onscreen(vec2 uv){
        return (uv.x < 1.) && (uv.x > 0.) && (uv.y < 1.) && (uv.y > 0.);
    }

    float border(vec2 uv, float border, vec2 texSize){
        uv*=texSize;
        return (uv.x<border || uv.x>texSize.x-border || uv.y<border || uv.y >texSize.y-border) ? 1.:.0;
    }
    SHADER_FS_INC

var shader_fs_copy = cat(shader_fs_inc, <<SHADER_FS_COPY);
    uniform sampler2D source;
    void main(void) {
        gl_FragColor = texture2D(source, uv);
    }
    SHADER_FS_COPY

var shader_fs_init = cat(shader_fs_inc, <<SHADER_FS_INIT);
    void main(void){
        gl_FragColor = encode2(vec2(0.));
    }
    SHADER_FS_INIT

var shader_fs_advance = cat(shader_fs_inc, <<SHADER_FS_ADVANCE);
    // Advance computes the next frame from the previous one (given computations made in other textures).
    // This is fed back into the simulation.  Contrast with 'composite', which is run before the final
    // display, but not fed back.
    uniform sampler2D sampler_prev;
    uniform sampler2D sampler_prev_n;
    uniform sampler2D sampler_blur;
    uniform sampler2D sampler_blur2;
    uniform sampler2D sampler_blur3;
    uniform sampler2D sampler_blur4;
    uniform sampler2D sampler_blur5;
    uniform sampler2D sampler_blur6;
    uniform sampler2D sampler_noise;
    uniform sampler2D sampler_noise_n;
    uniform sampler2D sampler_fluid;

    uniform vec4 rnd;
    uniform vec4 rainbow;
    uniform vec2 pixelSize;
    uniform vec2 aspect;
    uniform vec2 mouse;
    uniform vec2 mouseV;
    uniform float fps;

    float norm2(vec2 vin) {
        return vin.x*vin.x + vin.y*vin.y;
    }

    void main(void) {
        vec2 motion = decode2( texture2D(sampler_fluid, uv))*pixelSize*0.75;
        vec2 uv = uv - motion; // add fluid motion
        vec4 last = texture2D(sampler_prev, uv);

        float flatness = 0.0001;
        float addColor = 0.01 * flatness/(flatness + norm2(mouse - uv));
        float red = last.x - last.z + addColor;

        gl_FragColor = vec4(red, 0., -red, 1.);
    }
    SHADER_FS_ADVANCE

var shader_fs_composite = cat(shader_fs_inc, <<SHADER_FS_COMPOSITE);
    // Composite transforms the internal simulation state to a visible color -- it is not fed back
    // into the simulation.
    uniform sampler2D sampler_prev;
    uniform sampler2D sampler_prev_n;
    uniform sampler2D sampler_blur;
    uniform sampler2D sampler_blur2;
    uniform sampler2D sampler_blur3;
    uniform sampler2D sampler_blur4;
    uniform sampler2D sampler_blur5;
    uniform sampler2D sampler_blur6;
    uniform sampler2D sampler_noise;
    uniform sampler2D sampler_noise_n;
    uniform sampler2D sampler_fluid;
    uniform sampler2D sampler_fluid_p;

    uniform vec4 rnd;
    uniform vec4 rainbow;
    uniform vec2 pixelSize;
    uniform vec2 aspect;
    uniform vec2 mouse;
    uniform vec2 mouseV;
    uniform float fps;

    void main(void) {
        vec4 last = texture2D(sampler_prev, uv);
        gl_FragColor = last;
    }
    SHADER_FS_COMPOSITE

var shader_fs_blur_horizontal = cat(shader_fs_inc, <<SHADER_FS_BLUR_HORIZONTAL);
    // original shader from http://www.gamerendering.com/2008/10/11/gaussian-blur-filter-shader/
    // horizontal blur fragment shader
    uniform sampler2D src_tex;
    uniform vec2 pixelSize;

    void main(void) // fragment
    {
        float h = pixelSize.x;
        vec4 sum = vec4(0.0);
        sum += texture2D(src_tex, vec2(uv.x - 4.0*h, uv.y) ) * 0.05;
        sum += texture2D(src_tex, vec2(uv.x - 3.0*h, uv.y) ) * 0.09;
        sum += texture2D(src_tex, vec2(uv.x - 2.0*h, uv.y) ) * 0.12;
        sum += texture2D(src_tex, vec2(uv.x - 1.0*h, uv.y) ) * 0.15;
        sum += texture2D(src_tex, vec2(uv.x + 0.0*h, uv.y) ) * 0.16;
        sum += texture2D(src_tex, vec2(uv.x + 1.0*h, uv.y) ) * 0.15;
        sum += texture2D(src_tex, vec2(uv.x + 2.0*h, uv.y) ) * 0.12;
        sum += texture2D(src_tex, vec2(uv.x + 3.0*h, uv.y) ) * 0.09;
        sum += texture2D(src_tex, vec2(uv.x + 4.0*h, uv.y) ) * 0.05;
        gl_FragColor.xyz = sum.xyz/0.98; // normalize
        gl_FragColor.a = 1.;
    } 
    SHADER_FS_BLUR_HORIZONTAL

var shader_fs_composite = cat(shader_fs_inc, <<SHADER_FS_COMPOSITE);
    // Composite transforms the internal simulation state to a visible color -- it is not fed back
    // into the simulation.
    uniform sampler2D sampler_prev;
    uniform sampler2D sampler_prev_n;
    uniform sampler2D sampler_blur;
    uniform sampler2D sampler_blur2;
    uniform sampler2D sampler_blur3;
    uniform sampler2D sampler_blur4;
    uniform sampler2D sampler_blur5;
    uniform sampler2D sampler_blur6;
    uniform sampler2D sampler_noise;
    uniform sampler2D sampler_noise_n;
    uniform sampler2D sampler_fluid;
    uniform sampler2D sampler_fluid_p;

    uniform vec4 rnd;
    uniform vec4 rainbow;
    uniform vec2 pixelSize;
    uniform vec2 aspect;
    uniform vec2 mouse;
    uniform vec2 mouseV;
    uniform float fps;

    void main(void) {
        vec4 last = texture2D(sampler_prev, uv);
        gl_FragColor = last;
    }
    SHADER_FS_COMPOSITE

var shader_fs_blur_horizontal = cat(shader_fs_inc, <<SHADER_FS_BLUR_HORIZONTAL);
    // original shader from http://www.gamerendering.com/2008/10/11/gaussian-blur-filter-shader/
    // horizontal blur fragment shader
    uniform sampler2D src_tex;
    uniform vec2 pixelSize;

    void main(void) // fragment
    {
        float h = pixelSize.x;
        vec4 sum = vec4(0.0);
        sum += texture2D(src_tex, vec2(uv.x - 4.0*h, uv.y) ) * 0.05;
        sum += texture2D(src_tex, vec2(uv.x - 3.0*h, uv.y) ) * 0.09;
        sum += texture2D(src_tex, vec2(uv.x - 2.0*h, uv.y) ) * 0.12;
        sum += texture2D(src_tex, vec2(uv.x - 1.0*h, uv.y) ) * 0.15;
        sum += texture2D(src_tex, vec2(uv.x + 0.0*h, uv.y) ) * 0.16;
        sum += texture2D(src_tex, vec2(uv.x + 1.0*h, uv.y) ) * 0.15;
        sum += texture2D(src_tex, vec2(uv.x + 2.0*h, uv.y) ) * 0.12;
        sum += texture2D(src_tex, vec2(uv.x + 3.0*h, uv.y) ) * 0.09;
        sum += texture2D(src_tex, vec2(uv.x + 4.0*h, uv.y) ) * 0.05;
        gl_FragColor.xyz = sum.xyz/0.98; // normalize
        gl_FragColor.a = 1.;
    } 
    SHADER_FS_BLUR_HORIZONTAL

var shader_fs_blur_vertical = cat(shader_fs_inc, <<SHADER_FS_BLUR_VERTICAL);
    uniform sampler2D src_tex;
    uniform vec2 pixelSize;

    void main(void) // fragment
    {
        float v = pixelSize.y;
        vec4 sum = vec4(0.0);
        sum += texture2D(src_tex, vec2(uv.x, - 4.0*v + uv.y) ) * 0.05;
        sum += texture2D(src_tex, vec2(uv.x, - 3.0*v + uv.y) ) * 0.09;
        sum += texture2D(src_tex, vec2(uv.x, - 2.0*v + uv.y) ) * 0.12;
        sum += texture2D(src_tex, vec2(uv.x, - 1.0*v + uv.y) ) * 0.15;
        sum += texture2D(src_tex, vec2(uv.x, + 0.0*v + uv.y) ) * 0.16;
        sum += texture2D(src_tex, vec2(uv.x, + 1.0*v + uv.y) ) * 0.15;
        sum += texture2D(src_tex, vec2(uv.x, + 2.0*v + uv.y) ) * 0.12;
        sum += texture2D(src_tex, vec2(uv.x, + 3.0*v + uv.y) ) * 0.09;
        sum += texture2D(src_tex, vec2(uv.x, + 4.0*v + uv.y) ) * 0.05;
        gl_FragColor.xyz = sum.xyz/0.98;
        gl_FragColor.a = 1.;
    }
    SHADER_FS_BLUR_VERTICAL

var shader_fs_add_mouse_motion = cat(shader_fs_inc, <<SHADER_FS_ADD_MOUSE_MOTION);
    uniform sampler2D sampler_fluid;

    uniform vec2 aspect;
    uniform vec2 mouse; // mouse coordinate
    uniform vec2 mouseV; // mouse velocity
    uniform vec2 pixelSize;
    uniform vec2 texSize;

    float mouseFilter(vec2 uv){
        return clamp( 1.-length((uv-mouse)*texSize)/8., 0. , 1.);
    }

    void main(void){
        vec2 v = decode2(texture2D(sampler_fluid, uv));

        if(length(mouseV) > 0.)
            v = mix(v, mouseV, mouseFilter(uv)*0.85);

        gl_FragColor = encode2(v);
    }
    SHADER_FS_ADD_MOUSE_MOTION

var shader_fs_advect = cat(shader_fs_inc, <<SHADER_FS_ADVECT);
    uniform vec2 texSize;
    uniform vec2 pixelSize;
    uniform sampler2D sampler_fluid;

    const float dt = .001;

    void main(void){
        vec2 v = decode2(texture2D(sampler_fluid, uv));

        vec2 D = -texSize*vec2(v.x, v.y)*dt;

        vec2 Df = floor(D),   Dd = D - Df;
        vec2 uv = uv + Df*pixelSize;

        vec2 uv0, uv1, uv2, uv3;

        uv0 = uv + pixelSize*vec2(0.,0.);
        uv1 = uv + pixelSize*vec2(1.,0.);
        uv2 = uv + pixelSize*vec2(0.,1.);
        uv3 = uv + pixelSize*vec2(1.,1.);

        vec2 v0 = decode2( texture2D(sampler_fluid, uv0));
        vec2 v1 = decode2( texture2D(sampler_fluid, uv1));
        vec2 v2 = decode2( texture2D(sampler_fluid, uv2));
        vec2 v3 = decode2( texture2D(sampler_fluid, uv3));

        v = mix( mix( v0, v1, Dd.x), mix( v2, v3, Dd.x), Dd.y);

        gl_FragColor = encode2(v*(1.-border(uv, 1., texSize)));
    }
    SHADER_FS_ADVECT

var shader_fs_p = cat(shader_fs_inc, <<SHADER_FS_P);
    uniform vec2 pixelSize;
    uniform vec2 texSize;
    uniform sampler2D sampler_v;
    uniform sampler2D sampler_p;
    const float h = 1./1024.;

    void main(void){

        vec2 v = decode2(texture2D(sampler_v, uv));
        float v_x = decode(texture2D(sampler_v, uv - vec2(1.,0.)*pixelSize).rg);
        float v_y = decode(texture2D(sampler_v, uv - vec2(0.,1.)*pixelSize).ba);

        float n = decode(texture2D(sampler_p, uv- pixelSize*vec2(0.,1.)).rg);
        float w = decode(texture2D(sampler_p, uv + pixelSize*vec2(1.,0.)).rg);
        float s = decode(texture2D(sampler_p, uv + pixelSize*vec2(0.,1.)).rg);
        float e = decode(texture2D(sampler_p, uv - pixelSize*vec2(1.,0.)).rg);

        float p = ( n + w + s + e - (v.x - v_x + v.y - v_y)*h ) * .25;

        gl_FragColor.rg = encode(p);
        gl_FragColor.ba = vec2(0.); // unused
    }
    SHADER_FS_P

var shader_fs_div = cat(shader_fs_inc, <<SHADER_FS_DIV);
    uniform vec2 texSize;
    uniform vec2 pixelSize;
    uniform sampler2D sampler_v;
    uniform sampler2D sampler_p;

    void main(void){
        float p = decode(texture2D(sampler_p, uv).rg);
        vec2 v = decode2(texture2D(sampler_v, uv));
        float p_x = decode(texture2D(sampler_p, uv + vec2(1.,0.)*pixelSize).rg);
        float p_y = decode(texture2D(sampler_p, uv + vec2(0.,1.)*pixelSize).rg);

        v -= (vec2(p_x, p_y)-p)*512.;

        gl_FragColor = encode2(v);
    }
    SHADER_FS_DIV


// type is gl.FRAGMENT_SHADER or gl.VERTEX_SHADER
var makeShader = function(gl, type, code) {
    var shader = gl.createShader(type);
    gl.shaderSource(shader, code);
    gl.compileShader(shader);
    if (gl.getShaderParameter(shader, gl.COMPILE_STATUS) == 0) {
        console.log("error compiling shader", code, gl.getShaderInfoLog(shader));
    }
    return shader;
};
	
requestAnimFrame = 
    window.webkitRequestAnimationFrame
        || window.mozRequestAnimationFrame
        || function(callback, element) { setTimeout(callback, 1000 / 60); };


var gl;

var prog_copy;
var prog_advance;
var prog_composite;
var prog_blur_horizontal;
var prog_blur_vertical;

var prog_fluid_init;
var prog_fluid_add_mouse_motion;
var prog_fluid_advect;
var prog_fluid_p;
var prog_fluid_div;

var FBO_main;
var FBO_main2;

var FBO_noise;
var FBO_blur;
var FBO_blur2;
var FBO_blur3;
var FBO_blur4;
var FBO_blur5;
var FBO_blur6;

var FBO_helper;
var FBO_helper2;
var FBO_helper3;
var FBO_helper4;
var FBO_helper5;
var FBO_helper6;

var texture_main_l; // main, linear
var texture_main_n; // main, nearest (accurate uv access on the same buffer)
var texture_main2_l; // main double buffer, linear
var texture_main2_n; // main double buffer, nearest (accurate uv access on the same buffer)
var texture_helper; // needed for multi-pass shader programs (2-pass Gaussian blur)
var texture_helper2; // (1/4 resolution )
var texture_helper3; // (1/16 resolution )
var texture_helper4; // (1/256 resolution )
var texture_helper5;
var texture_helper6;

var texture_blur; // full resolution blur result
var texture_blur2; // double blur
var texture_blur3; // quad blur
var texture_blur4; // use low resolutions wisely ;)
var texture_blur5;
var texture_blur6;

var texture_noise_n; // nearest
var texture_noise_l; // linear interpolation

// fluid simulation GL textures and frame buffer objects

var texture_fluid_v;	// velocities
var texture_fluid_p;	// pressure
var texture_fluid_store;  
var texture_fluid_backbuffer;

var FBO_fluid_v;
var FBO_fluid_p;
var FBO_fluid_store;
var FBO_fluid_backbuffer;

var simScale = 2; // factor for reduced buffer size (TODO) 

// main animation loop vars

var sizeX = 1024;	// texture size (must be powers of two)
var sizeY = 512;

var viewX = sizeX;	// viewport size (ideally exactly the texture size)
var viewY = sizeY;

var halted = false;
var it = 1;	// main loop buffer toggle
var fps;

var mouseX = 0.5;
var mouseY = 0.5;
var oldMouseX = 0;
var oldMouseY = 0;
var mouseDx = 0;
var mouseDy = 0;

var load = function() {
    try {
        gl = canvas.getContext("experimental-webgl", {
            depth : false
        });
    } catch (e) {}
    if (!gl) {
        alert("Your browser does not support WebGL");
        return;
    }
    
    viewX = window.innerWidth;
    viewY = window.innerHeight;

    prog_copy = createAndLinkProgram(shader_fs_copy);
    
    prog_advance = createAndLinkProgram(shader_fs_advance);
    prog_composite = createAndLinkProgram(shader_fs_composite);
    prog_blur_horizontal = createAndLinkProgram(shader_fs_blur_horizontal);
    prog_blur_vertical = createAndLinkProgram(shader_fs_blur_vertical);
    
    prog_fluid_init = createAndLinkProgram(shader_fs_init); // sets encoded values to zero
    prog_fluid_add_mouse_motion = createAndLinkProgram(shader_fs_add_mouse_motion);
    prog_fluid_advect = createAndLinkProgram(shader_fs_advect);
    prog_fluid_p = createAndLinkProgram(shader_fs_p);
    prog_fluid_div = createAndLinkProgram(shader_fs_div);

    // two triangles ought to be enough for anyone ;)
    var posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);

    var vertices = new Float32Array([ -1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0 ]);

    var aPosLoc = gl.getAttribLocation(prog_advance, "aPos");
    gl.enableVertexAttribArray(aPosLoc);

    var aTexLoc = gl.getAttribLocation(prog_advance, "aTexCoord");
    gl.enableVertexAttribArray(aTexLoc);

    var texCoords = new Float32Array([ 0, 0, 1, 0, 0, 1, 1, 1 ]);

    var texCoordOffset = vertices.byteLength;

    gl.bufferData(gl.ARRAY_BUFFER, texCoordOffset + texCoords.byteLength, gl.STATIC_DRAW);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, vertices);
    gl.bufferSubData(gl.ARRAY_BUFFER, texCoordOffset, texCoords);
    gl.vertexAttribPointer(aPosLoc, 3, gl.FLOAT, gl.FALSE, 0, 0);
    gl.vertexAttribPointer(aTexLoc, 2, gl.FLOAT, gl.FALSE, 0, texCoordOffset);

    var noisePixels = [], pixels = [], simpixels = [], pixels2 = [], pixels3 = [], pixels4 = [], pixels5 = [], pixels6 = [];
    for ( var i = 0; i < sizeX; i++) {
        for ( var j = 0; j < sizeY; j++) {
            noisePixels.push(0,0,128,255);
            pixels.push(0, 0, 0, 255);
            if( i < sizeX/simScale && j < sizeY/simScale) simpixels.push(0, 0, 0, 255);
            if( i < sizeX/2 && j < sizeY/2) pixels2.push(0, 0, 0, 255);
            if( i < sizeX/4 && j < sizeY/4) pixels3.push(0, 0, 0, 255);
            if( i < sizeX/8 && j < sizeY/8) pixels4.push(0, 0, 0, 255);
            if( i < sizeX/16 && j < sizeY/16) pixels5.push(0, 0, 0, 255);
            if( i < sizeX/32 && j < sizeY/32) pixels6.push(0, 0, 0, 255);
        }
    }

    FBO_main = gl.createFramebuffer();
    FBO_main2 = gl.createFramebuffer();
    var glPixels;
    glPixels = new Uint8Array(noisePixels);
    texture_main_n = createAndBindTexture(glPixels, 1, FBO_main, gl.NEAREST);
    texture_main2_n = createAndBindTexture(glPixels, 1, FBO_main2, gl.NEAREST);
    glPixels = new Uint8Array(noisePixels);
    texture_main_l = createAndBindTexture(glPixels, 1, FBO_main, gl.LINEAR);
    texture_main2_l = createAndBindTexture(glPixels, 1, FBO_main2, gl.LINEAR);

    FBO_fluid_p = gl.createFramebuffer();
    FBO_fluid_v = gl.createFramebuffer();
    FBO_fluid_store = gl.createFramebuffer();
    FBO_fluid_backbuffer = gl.createFramebuffer();
    texture_fluid_v = createAndBindSimulationTexture(new Uint8Array(simpixels), FBO_fluid_v);
    texture_fluid_p = createAndBindSimulationTexture(new Uint8Array(simpixels), FBO_fluid_p);
    texture_fluid_store = createAndBindSimulationTexture(new Uint8Array(simpixels), FBO_fluid_store);
    texture_fluid_backbuffer = createAndBindSimulationTexture(new Uint8Array(simpixels), FBO_fluid_backbuffer);
    
    FBO_helper = gl.createFramebuffer();
    FBO_helper2 = gl.createFramebuffer();
    FBO_helper3 = gl.createFramebuffer();
    FBO_helper4 = gl.createFramebuffer();
    FBO_helper5 = gl.createFramebuffer();
    FBO_helper6 = gl.createFramebuffer();
    texture_helper = createAndBindTexture(new Uint8Array(pixels), 1, FBO_helper, gl.NEAREST); // helper buffers for the two-pass Gaussian blur calculation basically
    texture_helper2 = createAndBindTexture(new Uint8Array(pixels2), 2, FBO_helper2, gl.NEAREST);
    texture_helper3 = createAndBindTexture(new Uint8Array(pixels3), 4, FBO_helper3, gl.NEAREST);
    texture_helper4 = createAndBindTexture(new Uint8Array(pixels4), 8, FBO_helper4, gl.NEAREST);
    texture_helper5 = createAndBindTexture(new Uint8Array(pixels5), 16, FBO_helper5, gl.NEAREST);
    texture_helper6 = createAndBindTexture(new Uint8Array(pixels6), 32, FBO_helper6, gl.NEAREST);

    FBO_blur = gl.createFramebuffer();
    FBO_blur2 = gl.createFramebuffer();
    FBO_blur3 = gl.createFramebuffer();
    FBO_blur4 = gl.createFramebuffer();
    FBO_blur5 = gl.createFramebuffer();
    FBO_blur6 = gl.createFramebuffer();
    texture_blur = createAndBindTexture(new Uint8Array(pixels), 1, FBO_blur, gl.LINEAR);
    texture_blur2 = createAndBindTexture(new Uint8Array(pixels2), 2, FBO_blur2, gl.LINEAR);
    texture_blur3 = createAndBindTexture(new Uint8Array(pixels3), 4, FBO_blur3, gl.LINEAR);
    texture_blur4 = createAndBindTexture(new Uint8Array(pixels4), 8, FBO_blur4, gl.LINEAR);
    texture_blur5 = createAndBindTexture(new Uint8Array(pixels5), 16, FBO_blur5, gl.LINEAR);
    texture_blur6 = createAndBindTexture(new Uint8Array(pixels6), 32, FBO_blur6, gl.LINEAR);

    FBO_noise = gl.createFramebuffer();
    glPixels = new Uint8Array(noisePixels);
    texture_noise_n = createAndBindTexture(glPixels, 1, FBO_noise, gl.NEAREST);
    texture_noise_l = createAndBindTexture(glPixels, 1, FBO_noise, gl.LINEAR);

    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, texture_blur);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, texture_blur2);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, texture_blur3);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, texture_blur4);
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, texture_blur5);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, texture_blur6);
    gl.activeTexture(gl.TEXTURE8); gl.bindTexture(gl.TEXTURE_2D, texture_noise_l);
    gl.activeTexture(gl.TEXTURE9); gl.bindTexture(gl.TEXTURE_2D, texture_noise_n);
    gl.activeTexture(gl.TEXTURE10); gl.bindTexture(gl.TEXTURE_2D, texture_fluid_v);
    gl.activeTexture(gl.TEXTURE11); gl.bindTexture(gl.TEXTURE_2D, texture_fluid_p);

    calculateBlurTexture();

    fluidInit(FBO_fluid_v);
    fluidInit(FBO_fluid_p);
    fluidInit(FBO_fluid_store);
    fluidInit(FBO_fluid_backbuffer);
};


var createAndLinkProgram = function(fsCode){
    var program = gl.createProgram();
    gl.attachShader(program, makeShader(gl, gl.VERTEX_SHADER, shader_vs));
    gl.attachShader(program, makeShader(gl, gl.FRAGMENT_SHADER, fsCode));
    gl.linkProgram(program);
    return program;
};

	
var createAndBindTexture = function(glPixels, scale, fbo, filter) {
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, sizeX/scale, sizeY/scale, 0, gl.RGBA, gl.UNSIGNED_BYTE, glPixels);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    return texture;
};

var createAndBindSimulationTexture = function (glPixels, fbo) {
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, sizeX/simScale, sizeY/simScale, 0, gl.RGBA, gl.UNSIGNED_BYTE, glPixels);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S , gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T , gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    return texture;
};

var fluidInit = function(fbo) {
    gl.viewport(0, 0, sizeX/simScale, sizeY/simScale);
    gl.useProgram(prog_fluid_init);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();
};

var setUniforms = function(program) {
    gl.uniform4f(gl.getUniformLocation(program, "rnd"), Math.random(), Math.random(), Math.random(), Math.random());
    gl.uniform2f(gl.getUniformLocation(program, "texSize"), sizeX, sizeY);
    gl.uniform2f(gl.getUniformLocation(program, "pixelSize"), 1. / sizeX, 1. / sizeY);
    gl.uniform2f(gl.getUniformLocation(program, "aspect"), Math.max(1, viewX / viewY), Math.max(1, viewY / viewX));
    gl.uniform2f(gl.getUniformLocation(program, "mouse"), mouseX, mouseY);
    gl.uniform2f(gl.getUniformLocation(program, "mouseV"), mouseDx, mouseDy);
    gl.uniform1f(gl.getUniformLocation(program, "fps"), fps);

    gl.uniform1i(gl.getUniformLocation(program, "sampler_prev"), 0);
    gl.uniform1i(gl.getUniformLocation(program, "sampler_prev_n"), 1);
    gl.uniform1i(gl.getUniformLocation(program, "sampler_blur"), 2);
    gl.uniform1i(gl.getUniformLocation(program, "sampler_blur2"), 3);
    gl.uniform1i(gl.getUniformLocation(program, "sampler_blur3"), 4);
    gl.uniform1i(gl.getUniformLocation(program, "sampler_blur4"), 5);
    gl.uniform1i(gl.getUniformLocation(program, "sampler_blur5"), 6);
    gl.uniform1i(gl.getUniformLocation(program, "sampler_blur6"), 7);
    gl.uniform1i(gl.getUniformLocation(program, "sampler_noise"), 8);
    gl.uniform1i(gl.getUniformLocation(program, "sampler_noise_n"), 9);
    gl.uniform1i(gl.getUniformLocation(program, "sampler_fluid"), 10);
    gl.uniform1i(gl.getUniformLocation(program, "sampler_fluid_p"), 11);
};

var calculateBlurTextures = function() {
    var texture_source = (it < 0 ) ? texture_main2_l : texture_main_l;
    calculateBlurTexture(texture_source, texture_blur, FBO_blur, texture_helper, FBO_helper, 1);
    calculateBlurTexture(texture_blur, texture_blur2, FBO_blur2, texture_helper2, FBO_helper2, 2);
    calculateBlurTexture(texture_blur2, texture_blur3, FBO_blur3, texture_helper3, FBO_helper3, 4);
    calculateBlurTexture(texture_blur3, texture_blur4, FBO_blur4, texture_helper4, FBO_helper4, 8);
    calculateBlurTexture(texture_blur4, texture_blur5, FBO_blur5, texture_helper5, FBO_helper5, 16);
    calculateBlurTexture(texture_blur5, texture_blur6, FBO_blur6, texture_helper6, FBO_helper6, 32);
};

var calculateBlurTexture = function(sourceTex, targetTex, targetFBO, helperTex, helperFBO, scale) {
    // copy source
    gl.viewport(0, 0, sizeX / scale, sizeY / scale);
    gl.useProgram(prog_copy);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sourceTex);
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();

    // blur vertically
    gl.viewport(0, 0, sizeX / scale, sizeY / scale);
    gl.useProgram(prog_blur_vertical);
    gl.uniform2f(gl.getUniformLocation(prog_blur_vertical, "pixelSize"), scale / sizeX, scale / sizeY);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, targetTex);
    gl.bindFramebuffer(gl.FRAMEBUFFER, helperFBO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();

    // blur horizontally
    gl.viewport(0, 0, sizeX / scale, sizeY / scale);
    gl.useProgram(prog_blur_horizontal);
    gl.uniform2f(gl.getUniformLocation(prog_blur_horizontal, "pixelSize"), scale / sizeX, scale / sizeY);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, helperTex);
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();
};

var fluidSimulationStep = function() {
    addMouseMotion();
    advect();
    diffuse();
};

var addMouseMotion = function() {
    gl.viewport(0, 0, (sizeX/simScale), (sizeY/simScale));
    gl.useProgram(prog_fluid_add_mouse_motion);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture_fluid_v);
    gl.uniform2f(gl.getUniformLocation(prog_fluid_add_mouse_motion, "aspect"), Math.max(1, viewX / viewY), Math.max(1, viewY / viewX));
    gl.uniform2f(gl.getUniformLocation(prog_fluid_add_mouse_motion, "mouse"), mouseX, mouseY);
    gl.uniform2f(gl.getUniformLocation(prog_fluid_add_mouse_motion, "mouseV"), mouseDx, mouseDy);
    gl.uniform2f(gl.getUniformLocation(prog_fluid_add_mouse_motion, "pixelSize"), 1. / (sizeX/simScale), 1. / (sizeY/simScale));
    gl.uniform2f(gl.getUniformLocation(prog_fluid_add_mouse_motion, "texSize"), (sizeX/simScale), (sizeY/simScale));
    gl.bindFramebuffer(gl.FRAMEBUFFER, FBO_fluid_backbuffer);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();
};

var advect = function() {
    gl.viewport(0, 0, (sizeX/simScale), (sizeY/simScale));
    gl.useProgram(prog_fluid_advect);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture_fluid_backbuffer);
    gl.uniform2f(gl.getUniformLocation(prog_fluid_advect, "pixelSize"), 1. / (sizeX/simScale), 1. / (sizeY/simScale));
    gl.uniform2f(gl.getUniformLocation(prog_fluid_advect, "texSize"), (sizeX/simScale), (sizeY/simScale));
    gl.bindFramebuffer(gl.FRAMEBUFFER, FBO_fluid_v);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();
};

var diffuse = function() {
    for ( var i = 0; i < 8; i++) {
        gl.viewport(0, 0, (sizeX/simScale), (sizeY/simScale));
        gl.useProgram(prog_fluid_p);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture_fluid_v);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, texture_fluid_p);
        gl.uniform2f(gl.getUniformLocation(prog_fluid_p, "texSize"), (sizeX/simScale), (sizeY/simScale));
        gl.uniform2f(gl.getUniformLocation(prog_fluid_p, "pixelSize"), 1. / (sizeX/simScale), 1. / (sizeY/simScale));
        gl.uniform1i(gl.getUniformLocation(prog_fluid_p, "sampler_v"), 0);
        gl.uniform1i(gl.getUniformLocation(prog_fluid_p, "sampler_p"), 1);
        gl.bindFramebuffer(gl.FRAMEBUFFER, FBO_fluid_backbuffer);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.flush();

        gl.viewport(0, 0, (sizeX/simScale), (sizeY/simScale));
        gl.useProgram(prog_fluid_p);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture_fluid_v);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, texture_fluid_backbuffer);
        gl.uniform2f(gl.getUniformLocation(prog_fluid_p, "texSize"), (sizeX/simScale), (sizeY/simScale));
        gl.uniform2f(gl.getUniformLocation(prog_fluid_p, "pixelSize"), 1. / (sizeX/simScale), 1. / (sizeY/simScale));
        gl.uniform1i(gl.getUniformLocation(prog_fluid_p, "sampler_v"), 0);
        gl.uniform1i(gl.getUniformLocation(prog_fluid_p, "sampler_p"), 1);
        gl.bindFramebuffer(gl.FRAMEBUFFER, FBO_fluid_p);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.flush();
    }
    
    gl.viewport(0, 0, (sizeX/simScale), (sizeY/simScale));
    gl.useProgram(prog_fluid_div);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture_fluid_v);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, texture_fluid_p);
    gl.uniform2f(gl.getUniformLocation(prog_fluid_div, "texSize"), (sizeX/simScale), (sizeY/simScale));
    gl.uniform2f(gl.getUniformLocation(prog_fluid_div, "pixelSize"), 1. / (sizeX/simScale), 1. / (sizeY/simScale));
    gl.uniform1i(gl.getUniformLocation(prog_fluid_div, "sampler_v"), 0);
    gl.uniform1i(gl.getUniformLocation(prog_fluid_div, "sampler_p"), 1);
    gl.bindFramebuffer(gl.FRAMEBUFFER, FBO_fluid_v);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();
};

var advance = function() {
    fluidSimulationStep();

    // texture warp step

    gl.viewport(0, 0, sizeX, sizeY);
    gl.useProgram(prog_advance);
    setUniforms(prog_advance);
    if (it > 0) {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture_main_l); // interpolated input
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, texture_main_n); // "nearest" input
        gl.bindFramebuffer(gl.FRAMEBUFFER, FBO_main2); // write to buffer
    } else {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture_main2_l); // interpolated
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, texture_main2_n); // "nearest"
        gl.bindFramebuffer(gl.FRAMEBUFFER, FBO_main); // write to buffer
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();

    var pixelValues = new Uint8Array(4);
    gl.readPixels(sizeX/2, sizeY/2, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixelValues);
    $('#desc').text(pixelValues[0] + "," + pixelValues[1] + "," + pixelValues[2]);

    calculateBlurTextures();

    it = -it;
};

var composite = function() {
    gl.viewport(0, 0, viewX, viewY);
    gl.useProgram(prog_composite);
    setUniforms(prog_composite);
    if (it < 0) {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture_main_l);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, texture_main_n);
    } else {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture_main2_l);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, texture_main2_n);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();
};

var anim = function() {
    if (oldMouseX != 0 && oldMouseY != 0) {
        mouseDx = (mouseX - oldMouseX) * viewX;
        mouseDy = (mouseY - oldMouseY) * viewY;
    }

    if (!halted)
        advance();

    composite();

    setTimeout(function() { requestAnimFrame(anim) }, 0);

    oldMouseX = mouseX;
    oldMouseY = mouseY;
};


return {
    load: load,
    start: anim,
    setMouse: function(x,y) { mouseX = x; mouseY = y; }
}

};
