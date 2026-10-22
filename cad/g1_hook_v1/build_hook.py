"""Generate an editable concept hook, neutral CAD, and a MuJoCo fit fixture.

CAD units: mm. Model axes: X spans the mouth, Y is plate thickness/bar axis,
Z is upward. The mouth opens toward -Z. Origin is the seating-arc center.
Mount holes belong to the concept plate, NOT a measured G1 wrist interface.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import ezdxf
import mapbox_earcut
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vertices_faces(shape):
    return shape.vertices, shape.faces


def build(p):
    t = p['plate_thickness']; r = p['inner_radius']; width = p['section_width']
    assert p['bar_diameter'] > 0 and r > p['bar_diameter']/2
    assert min(t, width, p['front_lip_length'], p['back_leg_length']) > 0
    hole_r = p['concept_hole_diameter']/2
    assert 0 < hole_r < width/2
    assert 0 <= p['edge_chamfer'] < min(t/2, width/2-hole_r)
    assert int(p['collision_arc_segments']) >= 8
    for z in p['concept_hole_z']:
        assert -p['back_leg_length']+hole_r+p['edge_chamfer'] < z < -hole_r-p['edge_chamfer']
    assert all(abs(a-b) > 2*hole_r for i,a in enumerate(p['concept_hole_z']) for b in p['concept_hole_z'][i+1:])
    if p['edge_chamfer'] != 0:
        raise ValueError('This mesh builder supports unchamfered concept geometry only')
    outer=r+width; back=p['back_leg_length']; lip=p['front_lip_length']
    theta=np.linspace(np.pi,0,257)
    boundary=np.array([(-outer,-back)]+list(zip(outer*np.cos(theta),outer*np.sin(theta)))+
        [(outer,-lip),(r,-lip)]+list(zip(r*np.cos(theta[::-1]),r*np.sin(theta[::-1])))+[(-r,-back)])
    rings=[boundary]
    for z in p['concept_hole_z']:
        angle=np.linspace(0,2*np.pi,96,endpoint=False)
        rings.append(np.column_stack((-r-width/2+hole_r*np.cos(angle),z+hole_r*np.sin(angle))))
    points=np.concatenate(rings); n=len(points)
    triangles=mapbox_earcut.triangulate_float64(points,np.cumsum([len(ring) for ring in rings],dtype=np.uint32)).reshape(-1,3)
    vertices=np.array([(x,y,z) for y in (-t/2,t/2) for x,z in points])
    faces=triangles.tolist()+(triangles[:,::-1]+n).tolist()
    start=0
    for ring in rings:
        for i in range(len(ring)):
            a=start+i; b=start+(i+1)%len(ring)
            faces.extend([[a,b,b+n],[a,b+n,a+n]])
        start+=len(ring)
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True)
    mesh.fix_normals()
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.body_count==1
    return mesh


def export_editable(p,out):
    r=p['inner_radius']; outer=r+p['section_width']; back=p['back_leg_length']; lip=p['front_lip_length']
    # Analytic DXF: two true semicircular arcs, straight sides and round holes.
    doc=ezdxf.new('R2010'); doc.units=4; space=doc.modelspace()
    space.add_lwpolyline([(-outer,-back,0),(-outer,0,-1),(outer,0,0),(outer,-lip,0),
        (r,-lip,0),(r,0,1),(-r,0,0),(-r,-back,0)],format='xyb',close=True)
    for z in p['concept_hole_z']:
        space.add_circle((-r-p['section_width']/2,z),p['concept_hole_diameter']/2)
    doc.saveas(out/'hook_profile_mm.dxf')
    source='''// Units mm. Concept only. Holes do not define the G1 wrist interface.
// X=mouth width; Y=plate thickness and bar axis; Z=up; opening toward -Z.
$fn = 256;
inner_radius = INNER;
section_width = WIDTH;
plate_thickness = THICK;
back_leg_length = BACK;
front_lip_length = LIP;
hole_diameter = HOLE;
hole_z = HOLES;
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
'''
    for token,key in [('INNER','inner_radius'),('WIDTH','section_width'),('THICK','plate_thickness'),
                      ('BACK','back_leg_length'),('LIP','front_lip_length'),('HOLES','concept_hole_z'),('HOLE','concept_hole_diameter')]:
        source=source.replace(token,json.dumps(p[key]))
    (out/'hook.scad').write_text(source,encoding='utf-8')


def mujoco_fixture(p, shape, out, center, inertia, mass):
    r = p['inner_radius']/1000; outer = r+p['section_width']/1000
    half_t = p['plate_thickness']/2000; n = int(p['collision_arc_segments'])
    root = ET.Element('mujoco', model='G1 hook CAD v1 - geometry inspection only')
    ET.SubElement(root, 'compiler', angle='radian', meshdir='.', autolimits='true')
    ET.SubElement(root, 'option', timestep='.0005', gravity='0 0 -9.81')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1200', offheight='900')
    assets = ET.SubElement(root, 'asset')
    ET.SubElement(assets, 'mesh', name='hook_visual', file='hook_mm.stl', scale='.001 .001 .001')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='.1 -.2 .4', dir='-.2 .4 -1')
    ET.SubElement(world, 'geom', name='test_bar', type='capsule',
        fromto='0 -.08 0 0 .08 0', size=str(p['bar_diameter']/2000), rgba='.95 .53 .14 1')
    hook = ET.SubElement(world, 'body', name='hook_cad')
    ET.SubElement(hook, 'freejoint', name='hook_free')
    vals = [inertia[0,0], inertia[1,1], inertia[2,2], inertia[0,1], inertia[0,2], inertia[1,2]]
    ET.SubElement(hook, 'inertial', pos=' '.join(map(str, center)), mass=str(mass), fullinertia=' '.join(map(str, vals)))
    ET.SubElement(hook, 'geom', name='hook_visual', mesh='hook_visual', type='mesh',
        contype='0', conaffinity='0', group='2', rgba='.13 .52 .69 1', mass='0')
    common = dict(group='3', rgba='.1 .8 .2 .3', mass='0', solref='.004 1', friction='.6 .01 .001')
    for i in range(n):
        a, b = math.pi*i/n, math.pi*(i+1)/n
        polygon = [(outer*math.cos(a), outer*math.sin(a)),
                   (outer*math.cos(b), outer*math.sin(b)),
                   (r*math.cos(b), r*math.sin(b)), (r*math.cos(a), r*math.sin(a))]
        vertices = [(x,y,z) for y in (-half_t, half_t) for x,z in polygon]
        name = f'arc_{i:02d}'
        ET.SubElement(assets, 'mesh', name=name, vertex=' '.join(str(v) for point in vertices for v in point))
        ET.SubElement(hook, 'geom', name=name, mesh=name, type='mesh', **common)
    for name, x, length in (('back', -(r+outer)/2, p['back_leg_length']/1000),
                             ('lip', (r+outer)/2, p['front_lip_length']/1000)):
        ET.SubElement(hook, 'geom', name=name, type='box', pos=f'{x} 0 {-length/2}',
            size=f'{(outer-r)/2} {half_t} {length/2}', **common)
    ET.SubElement(hook, 'site', name='seat_top', pos=f'0 0 {r}', size='.001')
    ET.indent(root)
    ET.ElementTree(root).write(out/'hook_fit.xml', encoding='utf-8')


def preview(p, shape, out, mass):
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(14,8), facecolor='#f4f7fa')
    fig.suptitle('G1 开口式钩爪 · 参数化概念模型 V1', fontsize=22, x=.055, ha='left', y=.96, color='#162e41')
    fig.text(.055,.902,'用于尺寸、装配与接触检查；腕部转接件待设计，尚无承载等级。', fontsize=12, color='#536879')
    ax = fig.add_axes([.025,.16,.50,.70], projection='3d', facecolor='#f4f7fa')
    v, f = vertices_faces(shape)
    ax.add_collection3d(Poly3DCollection(v[f], facecolors='#2588aa', linewidths=0, shade=True,
                                        lightsource=matplotlib.colors.LightSource(azdeg=225,altdeg=40)))
    # Seated rod center is below the inner crown by one bar radius.
    radius = p['bar_diameter']/2; zcenter = p['inner_radius']-radius
    angle = np.linspace(0,2*np.pi,64); length=np.linspace(-48,48,2)
    aa,ll=np.meshgrid(angle,length)
    ax.plot_surface(radius*np.cos(aa),ll,zcenter+radius*np.sin(aa),color='#ec983c',alpha=.85,shade=True)
    ax.set(xlim=(-55,55),ylim=(-55,55),zlim=(-62,48))
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=18,azim=-66); ax.set_axis_off()
    ax.text2D(.15,.02,'青色：钩体   橙色：Ø30 mm 横杆', transform=ax.transAxes,fontsize=11,color='#334d5f')
    side = fig.add_axes([.58,.26,.36,.58],facecolor='#f4f7fa')
    r=p['inner_radius']; R=r+p['section_width']; back=p['back_leg_length']; lip=p['front_lip_length']
    theta=np.linspace(np.pi,0,150)
    points=[(-R,-back),(-R,0)]+list(zip(R*np.cos(theta),R*np.sin(theta)))+[(R,-lip),(r,-lip),(r,0)]
    points+=list(zip(r*np.cos(theta[::-1]),r*np.sin(theta[::-1])))+[(-r,-back)]
    side.add_patch(Polygon(points,facecolor='#2588aa',edgecolor='#16526b',linewidth=1.5))
    for z in p['concept_hole_z']:
        side.add_patch(Circle((-r-p['section_width']/2,z),p['concept_hole_diameter']/2,facecolor='#f4f7fa',edgecolor='#16526b'))
    side.add_patch(Circle((0,zcenter),radius,facecolor='#ec983c',edgecolor='#aa6420',alpha=.8))
    side.annotate('',xy=(-r,-19),xytext=(r,-19),arrowprops=dict(arrowstyle='<->',color='#243e50'))
    side.text(0,-24,f'开口 {2*r:g} mm',ha='center',va='top',fontsize=11)
    side.annotate('',xy=(0,-2),xytext=(0,-52),arrowprops=dict(arrowstyle='->',color='#b77022',lw=2))
    side.text(4,-49,'横杆相对进入方向',fontsize=10,color='#9d5e1a')
    side.text(0,42,f'外宽 {2*R:g} mm  /  总高 {back+R:g} mm',ha='center',fontsize=11)
    side.text(0,33,f'承载圆弧 R{r:g}',ha='center',va='top',fontsize=10,color='white')
    side.set(xlim=(-46,46),ylim=(-64,48)); side.set_aspect('equal'); side.axis('off')
    fig.text(.57,.20,f'板厚 {p["plate_thickness"]:g} mm  |  短端长 {lip:g} mm  |  边缘倒角待定',fontsize=11,color='#334d5f')
    fig.text(.57,.155,f'2 × Ø{p["concept_hole_diameter"]:g} 接口占位孔：非 G1 官方安装孔位',fontsize=11,color='#9b5b23')
    fig.text(.055,.09,f'几何质量估计 {mass*1000:.1f} g（密度假设 7850 kg/m³；不含腕部转接件和紧固件）',fontsize=12,color='#334d5f')
    fig.text(.055,.045,'交付：参数 JSON + OpenSCAD 源码 + DXF 轮廓 + 封闭 STL 网格 + MuJoCo 检查场景',fontsize=11,color='#536879')
    fig.savefig(out/'preview.png',dpi=160,facecolor=fig.get_facecolor()); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--params',type=Path,default=HERE/'parameters.json')
    parser.add_argument('--out',type=Path,default=HERE/'model')
    a=parser.parse_args(); p=json.loads(a.params.read_text(encoding='utf-8'))
    if a.out.exists():
        raise FileExistsError('Use a new output directory to preserve earlier CAD revisions')
    shape=build(p); a.out.mkdir(parents=True)
    shape.export(a.out/'hook_mm.stl')
    export_editable(p,a.out)
    volume=shape.volume; center=shape.center_mass/1000
    density=p['assumed_density_kg_m3']; mass=volume*1e-9*density
    inertia=shape.moment_inertia*density*1e-15
    r=p['inner_radius']; outer=r+p['section_width']
    analytic_volume=(np.pi/2*(outer**2-r**2)+p['section_width']*(p['back_leg_length']+p['front_lip_length'])
        -len(p['concept_hole_z'])*np.pi*(p['concept_hole_diameter']/2)**2)*p['plate_thickness']
    assert abs(volume-analytic_volume)/analytic_volume < 1e-4
    mesh=trimesh.load_mesh(a.out/'hook_mm.stl',process=True)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
    assert abs(mesh.volume-volume)/volume < .002
    assert np.linalg.eigvalsh(inertia).min()>0
    mujoco_fixture(p,shape,a.out,center,inertia,mass)
    preview(p,shape,a.out,mass)
    metrics=dict(revision=p['revision'],units='CAD mm; MuJoCo m',trimesh=trimesh.__version__,
        mesh_valid=True,solid_count=1,stl_watertight=bool(mesh.is_watertight),
        step_exported=False,step_limitation='CadQuery OCP DLL blocked by Windows application control policy',
        openscad_render_tested=False,dxf_readback_entities=len(ezdxf.readfile(a.out/'hook_profile_mm.dxf').modelspace()),
        analytic_volume_mm3=analytic_volume,mesh_analytic_volume_relative_error=abs(volume-analytic_volume)/analytic_volume,
        stl_volume_relative_error=abs(mesh.volume-volume)/volume,
        volume_mm3=volume,mass_kg=mass,assumed_density_kg_m3=density,
        center_of_mass_m=center.tolist(),inertia_about_com_kg_m2=inertia.tolist(),
        opening_mm=2*p['inner_radius'],bar_diameter_mm=p['bar_diameter'],
        nominal_diametral_clearance_mm=2*p['inner_radius']-p['bar_diameter'],
        collision_model='24 convex arc wedges plus two boxes; no concave single-mesh collider',
        collision_approximations='Mount holes omitted; ring arcs polygonized; no chamfer in this revision',
        max_inner_arc_radial_approximation_mm=p['inner_radius']*(1-math.cos(math.pi/(2*p['collision_arc_segments']))),
        real_wrist_fit_verified=False,strength_verified=False,dynamic_hang_verified=False,
        repository_reference='https://github.com/ccrpRepo/RoboMimic_Deploy',
        active_checkout_origin='https://github.com/lu-yidan/RoboMimic_Deploy.git',
        source_sha256=sha(__file__),parameters_sha256=sha(a.params))
    (a.out/'parameters.json').write_bytes(a.params.read_bytes())
    (a.out/'build_hook.py').write_bytes(Path(__file__).read_bytes())
    metrics['files_sha256']={f.name:sha(f) for f in a.out.iterdir() if f.is_file()}
    (a.out/'manifest.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps({k:metrics[k] for k in ('mesh_valid','solid_count','stl_watertight','mass_kg','opening_mm','nominal_diametral_clearance_mm')},indent=2))


if __name__=='__main__':
    main()
