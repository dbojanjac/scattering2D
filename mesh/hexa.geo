n = 4;

/*********************************************************************
 *
 *  HEXA square unit cell in 2D inside circle
 *
 *********************************************************************/

 // Physical Surface(1) = Inner Hexagonal Structure (Air)
 // Physical Surface(2) = Inner Box containing Hexagonal material (Si)
 // Physical Surface(3) = Outer Box (Air)

// parameters
a = 0.4;
kor = 0;
v = a * Sqrt( 3 ) / 2;

alt = 1;

// Scaling parameters
h = 1 / n;

// Meshing parameters
lc1 = 3 * 1e-2 * h;
lc2 = 5 * 1e-2;
lc3 = 8 * 1e-2;


// Macro defining unit cell
Macro HEXA

 p1 = newp; Point(p1) = {h*(x0 - a/2) + t1 * h, h*(y0 - v + kor) + t2 * h + alt * h/2, 0, lc1};
 p2 = newp; Point(p2) = {h*(x0 + a/2) + t1 * h, h*(y0 - v + kor) + t2 * h + alt * h/2, 0, lc1};
 p3 = newp; Point(p3) = {h*(x0 + a) + t1 * h,   h*(y0) + t2 * h + alt * h/2,           0, lc1};
 p4 = newp; Point(p4) = {h*(x0 + a/2) + t1 * h, h*(y0 + v - kor) + t2 * h + alt * h/2, 0, lc1};
 p5 = newp; Point(p5) = {h*(x0 - a/2) + t1 * h, h*(y0 + v - kor) + t2 * h + alt * h/2, 0, lc1};
 p6 = newp; Point(p6) = {h*(x0 - a) + t1 * h,   h*(y0) + t2 * h + alt * h/2,           0, lc1};

 l1 = newl; Line(l1) = {p1, p2};
 l2 = newl; Line(l2) = {p2, p3};
 l3 = newl; Line(l3) = {p3, p4};
 l4 = newl; Line(l4) = {p4, p5};
 l5 = newl; Line(l5) = {p5, p6};
 l6 = newl; Line(l6) = {p6, p1};

 ll1 = newreg; Line Loop(ll1) = {l1, l2, l3, l4, l5, l6}; Plane Surface(ll1) = {ll1};
 loops[t1 * n + t2] = ll1;

 ll2 = newreg; Line Loop(ll2) = {l6, l5, l4, l3, l2, l1};
 loopsOpp[t1 * n + t2] = ll2;

Return

Color Red{ Surface{ 1 }; }
x0 = .5; y0 = 0.5;

For t1 In {0 : n - 1}
 alt = t1 % 2;

 For t2 In {0 : n - 1}
  Call HEXA;

 EndFor
EndFor

// All hexa are the same surface
Physical Surface(1) = {loops[]};

// Outer material Box
p00 = newp; Point(p00) = {0, 0,         0, lc2};
p10 = newp; Point(p10) = {1, 0,         0, lc2};
p01 = newp; Point(p01) = {1, 1 + h / 2, 0, lc2};
p11 = newp; Point(p11) = {0, 1 + h / 2, 0, lc2};

l00 = newl; Line(l00) = {p00, p10};
l11 = newl; Line(l11) = {p10, p01};
l22 = newl; Line(l22) = {p01, p11};
l33 = newl; Line(l33) = {p11, p00};

llBox = newreg; Line Loop(llBox) = {l00, l11, l22, l33};
Plane Surface (llBox) = {llBox, loopsOpp[]};

Physical Surface(2) = {llBox};

// Outer circle
pc0 = newp; Point(pc0) = {0.5,  0.5,  0, lc3};
pc1 = newp; Point(pc1) = {6.5,  0.5,  0, lc3};
pc2 = newp; Point(pc2) = {0.5,  6.5,  0, lc3};
pc3 = newp; Point(pc3) = {-5.5, 0.5,  0, lc3};
pc4 = newp; Point(pc4) = {0.5,  -5.5, 0, lc3};

c1 = newl; Circle(c1) = {pc1, pc0, pc2};
c2 = newl; Circle(c2) = {pc2, pc0, pc3};
c3 = newl; Circle(c3) = {pc3, pc0, pc4};
c4 = newl; Circle(c4) = {pc4, pc0, pc1};

llc_Box = newreg; Line Loop(llc_Box) = {c1, c2, c3, c4};
Plane Surface(llc_Box) = {llc_Box, llBox};
Physical Surface(3) = {llc_Box};


Color Red{ Surface{ loops[] }; }
