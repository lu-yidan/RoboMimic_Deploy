// Units mm. Concept only. Holes do not define the G1 wrist interface.
// X=mouth width; Y=plate thickness and bar axis; Z=up; opening toward -Z.
$fn = 256;
inner_radius = 22.0;
section_width = 12.0;
plate_thickness = 8.0;
back_leg_length = 55.0;
front_lip_length = 12.0;
hole_diameter = 5.5;
hole_z = [-44.0, -28.0];
outer_radius = inner_radius + section_width;
module profile() {
  difference() {
    union() {
      intersection() {
        difference() { circle(r=outer_radius); circle(r=inner_radius); }
        translate([-outer_radius,0]) square([2*outer_radius,outer_radius]);
      }
      translate([-outer_radius,-back_leg_length]) square([section_width,back_leg_length]);
      translate([inner_radius,-front_lip_length]) square([section_width,front_lip_length]);
    }
    for(z=hole_z) translate([-inner_radius-section_width/2,z]) circle(d=hole_diameter);
  }
}
rotate([90,0,0]) linear_extrude(height=plate_thickness,center=true) profile();
