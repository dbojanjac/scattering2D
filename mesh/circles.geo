n = 10;
r = 0.3;
lc1 = 2e-2;
lc2 = 2e-2;
lc3 = 2e-2;

// Macro defining unit cell
// n = 2
// center_x = {1/4 + 0, 1/4 + 1/2}
// center_y = {1/4, 1/4 + 1/2}

// n = 3
// center_x = {1/6, 1/3 + 1/6, 2/3 + 1/6}

Macro CIRCLE
    center_x = t1 / n + 1/(2*n);
    center_y = t2 / n + 1/(2*n);
    radius = r / n;
    p1 = newp; Point(p1) = {center_x,          center_y,          0, lc1};
    p2 = newp; Point(p2) = {center_x + radius, center_y,          0, lc1};
    p3 = newp; Point(p3) = {center_x,          center_y + radius, 0, lc1};
    p4 = newp; Point(p4) = {center_x - radius, center_y,          0, lc1};
    p5 = newp; Point(p5) = {center_x,          center_y - radius, 0, lc1};

    l1 = newl; Circle(l1) = {p2, p1, p3};
    l2 = newl; Circle(l2) = {p3, p1, p4};
    l3 = newl; Circle(l3) = {p4, p1, p5};
    l4 = newl; Circle(l4) = {p5, p1, p2};

    ll1 = newreg; Line Loop(ll1) = {l1, l2, l3, l4};
    Plane Surface(ll1) = {ll1};
    loops[t1 * n + t2] = ll1;

    ll2 = newreg; Line Loop(ll2) = {l4, l3, l2, l1};
    loopsOpp[t1 * n + t2] = ll2;

Return

Color Red{ Surface{ 1 }; }

For t1 In {0 : n-1}
    For t2 In {0 : n-1}
        Call CIRCLE;
    EndFor
EndFor

// All circles are the same surface
Physical Surface(1) = {loops[]};

// Outer material Box
p00 = newp; Point(p00) = {0, 0, 0, lc2};
p10 = newp; Point(p10) = {1, 0, 0, lc2};
p01 = newp; Point(p01) = {1, 1, 0, lc2};
p11 = newp; Point(p11) = {0, 1, 0, lc2};

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
