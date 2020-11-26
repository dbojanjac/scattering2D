n = 6;

/*********************************************************************
 *
 *  HEXA square unit cell in 2D inside circle
 *
 *********************************************************************/

// parameters
a = 0.4;
kor = 0;
v = a * Sqrt( 3 ) / 2;

// Extrude Height
hg = 0.1;
alt = 1;

// Scaling parameter
h = 1 / n;

// Meshing parameters
lc1 = 20 * 1e-2 * h;
lc2 = 120 * 1e-2;


// Macro defining unit cell
Macro HEXA

 // Points in (x, y, 0) Plane
 p1 = newp; Point(p1) = {h*(x0 - a/2) + t1 * h -0.5, h*(y0 - v + kor) + t2 * h + alt * h/2 -0.5, -0.5, lc1};
 p2 = newp; Point(p2) = {h*(x0 + a/2) + t1 * h -0.5, h*(y0 - v + kor) + t2 * h + alt * h/2 -0.5, -0.5, lc1};
 p3 = newp; Point(p3) = {h*(x0 + a) + t1 * h -0.5,   h*(y0) + t2 * h + alt * h/2 -0.5,           -0.5, lc1};
 p4 = newp; Point(p4) = {h*(x0 + a/2) + t1 * h -0.5, h*(y0 + v - kor) + t2 * h + alt * h/2 -0.5, -0.5, lc1};
 p5 = newp; Point(p5) = {h*(x0 - a/2) + t1 * h -0.5, h*(y0 + v - kor) + t2 * h + alt * h/2 -0.5, -0.5, lc1};
 p6 = newp; Point(p6) = {h*(x0 - a) + t1 * h -0.5,   h*(y0) + t2 * h + alt * h/2 -0.5,           -0.5, lc1};


 // Lines in (x, y, 0) Plane
 l1 = newl; Line(l1) = {p1, p2};
 l2 = newl; Line(l2) = {p2, p3};
 l3 = newl; Line(l3) = {p3, p4};
 l4 = newl; Line(l4) = {p4, p5};
 l5 = newl; Line(l5) = {p5, p6};
 l6 = newl; Line(l6) = {p6, p1};

 // Line loop in (x, y, 0) Plane
 ll = newreg; Line Loop(ll) = {l1, l2, l3, l4, l5, l6};
 lloops[t1 * n + t2] = ll;

 // Surface in (x, y, 0) Plane
 s = news; Plane Surface(s) = {ll};
 surfaces[t1 * n + t2] = s;

Return

// CALL MACRO
x0 = .5; y0 = 0.5;

For t1 In {0 : n - 1}
 alt = t1 % 2;

 For t2 In {0 : n - 1}
  Call HEXA;

 EndFor
EndFor


// Surface in (x, y, 0) Plane
s = news; Plane Surface(s) = {lloops[]};


Color Blue{ Surface{ surfaces[] }; }


// Si Material Box

// Si Points in (x, y, 0) Plane
p10 = newp; Point(p10) = {-0.5, -0.5,         -0.5, lc1};
p20 = newp; Point(p20) = {0.5, -0.5,         -0.5, lc1};
p30 = newp; Point(p30) = {0.5, 0.5 + h / 2, -0.5, lc1};
p40 = newp; Point(p40) = {-0.5, 0.5 + h / 2, -0.5, lc1};

// Si Lines in (x, y, 0) Plane
l10 = newl; Line(l10) = {p10, p20};
l20 = newl; Line(l20) = {p20, p30};
l30 = newl; Line(l30) = {p30, p40};
l40 = newl; Line(l40) = {p40, p10};

// Si Line Loop in (x, y, 0) Plane
llSi = newreg; Line Loop(llSi) = {l10, l20, l30, l40};

// Surface in (x, y, 0) Plane
sSi = news; Plane Surface(sSi) = {llSi, -lloops[]};
Color Red{ Surface{ sSi }; }


For t1 In {0 : n - 1}
 For t2 In {0 : n - 1}
  out[] = Extrude{0, 0, hg}{ Surface{ surfaces[t1 * n + t2] }; };
  phy[t1 * n + t2] = out[1];
  top[t1 * n + t2] = out[0];

 EndFor
EndFor



outSi[] = Extrude{0, 0, hg}{ Surface{ sSi }; };

Coherence Mesh;

//Outer surface Loop
osl = newreg; Surface Loop(osl) = {top[], surfaces[], sSi, outSi[0], outSi[2], outSi[3], outSi[4], outSi[5]};


Color Blue{ Volume{ phy[] }; }
Color Yellow{ Volume{ outSi[1] }; }

// Volume 1 is air
Physical Volume(1) = {phy[]};

// Volume 2 is Si
Physical Volume(2) = {outSi[1]};


// Outer Box
x = 0;
y = 0;
z = 0;
r = 4;

p1 = newp; Point(p1) = {x,  y,  z,   lc2} ;
p2 = newp; Point(p2) = {x+r,y,  z,   lc2} ;
p3 = newp; Point(p3) = {x,  y+r,z,   lc2} ;
p4 = newp; Point(p4) = {x,  y,  z+r, lc2} ;
p5 = newp; Point(p5) = {x-r,y,  z,   lc2} ;
p6 = newp; Point(p6) = {x,  y-r,z,   lc2} ;
p7 = newp; Point(p7) = {x,  y,  z-r, lc2} ;

c1 = newreg; Circle(c1) = {p2,p1,p7}; c2 = newreg; Circle(c2) = {p7,p1,p5};
c3 = newreg; Circle(c3) = {p5,p1,p4}; c4 = newreg; Circle(c4) = {p4,p1,p2};
c5 = newreg; Circle(c5) = {p2,p1,p3}; c6 = newreg; Circle(c6) = {p3,p1,p5};
c7 = newreg; Circle(c7) = {p5,p1,p6}; c8 = newreg; Circle(c8) = {p6,p1,p2};
c9 = newreg; Circle(c9) = {p7,p1,p3}; c10 = newreg; Circle(c10) = {p3,p1,p4};
c11 = newreg; Circle(c11) = {p4,p1,p6}; c12 = newreg; Circle(c12) = {p6,p1,p7};

l1 = newreg; Line Loop(l1) = {c5,c10,c4};    Surface(newreg) = {l1};
l2 = newreg; Line Loop(l2) = {c9,-c5,c1};    Surface(newreg) = {l2};
l3 = newreg; Line Loop(l3) = {c12,-c8,-c1};  Surface(newreg) = {l3};
l4 = newreg; Line Loop(l4) = {c8,-c4,c11};   Surface(newreg) = {l4};
l5 = newreg; Line Loop(l5) = {-c10,c6,c3};   Surface(newreg) = {l5};
l6 = newreg; Line Loop(l6) = {-c11,-c3,c7};  Surface(newreg) = {l6};
l7 = newreg; Line Loop(l7) = {-c2,-c7,-c12}; Surface(newreg) = {l7};
l8 = newreg; Line Loop(l8) = {-c6,-c9,c2};   Surface(newreg) = {l8};

loop = newreg; Surface Loop(loop) = {l8 + 1, l5 + 1, l1 + 1, l2 + 1, l3 + 1, l7 + 1, l6 + 1, l4 + 1};
slmb = newreg; Volume(slmb) = {loop, osl};

Coherence Mesh;

// Volume 3 is outer box air
Physical Volume(3) = slmb;
