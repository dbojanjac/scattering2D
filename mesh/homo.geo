//n = 10;

/*********************************************************************
 *
 *  HOMO square unit cell in 2D inside circle
 *
 *********************************************************************/

 // Physical Surface(1) = Inner Hexagonal Structure (Air)
 // Physical Surface(2) = Outer Box (Air)

// parameters
a = 0.4;
kor = 0;
v = a * Sqrt( 3 ) / 2;

alt = 1;

// Scaling parameters
h = 1 / n;

// Meshing parameters
lc1 = 1 * 1e-3;
lc2 = 4 * 1e-2;
lc3 = 1 * 1e-1;

// Outer material Box
p00 = newp; Point(p00) = {0, 0,         0, lc2};
p10 = newp; Point(p10) = {1, 0,         0, lc2};
p01 = newp; Point(p01) = {1, 1 + h / 2, 0, lc2};
p11 = newp; Point(p11) = {0, 1 + h / 2, 0, lc2};

m00 = newp; Point(m00) = {0.5, 0.5, 0, lc1};

l00 = newl; Line(l00) = {p00, p10};
l11 = newl; Line(l11) = {p10, p01};
l22 = newl; Line(l22) = {p01, p11};
l33 = newl; Line(l33) = {p11, p00};

llBox = newreg; Line Loop(llBox) = {l00, l11, l22, l33};
Plane Surface (llBox) = {llBox};
Point {m00} In Surface{llBox};

Physical Surface(1) = {llBox};

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
Physical Surface(2) = {llc_Box};
