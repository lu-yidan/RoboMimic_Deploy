"""Black modular hook: actual multipart CAD mesh and dimensioned previews.

Millimetres internally, metres in GLB and MuJoCo. No V1 source/asset mutation.
Mesh components remain separate; fasteners are simplified smooth proxies.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import mapbox_earcut
import trimesh
from trimesh.visual.material import PBRMaterial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource

HERE=Path(__file__).resolve().parent


def extrude(rings,thickness,y=0):
    points=np.concatenate(rings); n=len(points)
    faces=mapbox_earcut.triangulate_float64(points,np.cumsum([len(r) for r in rings],dtype=np.uint32)).reshape(-1,3)
    faces=faces.tolist()+(faces[:,::-1]+n).tolist()
    start=0
    for ring in rings:
        for i in range(len(ring)):
            a=start+i;b=start+(i+1)%len(ring)
            faces.extend([[a,b,b+n],[a,b+n,a+n]])
        start+=len(ring)
    vertices=np.array([(x,yy,z) for yy in (y-thickness/2,y+thickness/2) for x,z in points])
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True)
    mesh.fix_normals()
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
    return mesh


def circle(x,z,r,n=64):
    a=np.linspace(0,2*np.pi,n,endpoint=False)
    return np.column_stack((x+r*np.cos(a),z+r*np.sin(a)))


def sector(inner,outer,a,b,n=128):
    angle=np.linspace(a,b,n)
    return np.concatenate((np.column_stack((outer*np.cos(angle),outer*np.sin(angle))),
                           np.column_stack((inner*np.cos(angle[::-1]),inner*np.sin(angle[::-1])))))


def create(p):
    r=p['plate_inner_radius'];R=p['plate_outer_radius'];t=p['plate_thickness'];gap=p['plate_inner_gap']
    back=p['back_length'];lip=p['lip_length'];clip=p['corner_clip'];hole=p['hole_diameter']/2
    assert p['bar_diameter']/2 < p['liner_inner_radius'] < r < p['liner_outer_radius']
    assert p['liner_outer_radius']==p['carrier_inner_radius'] < p['carrier_outer_radius']==R
    assert t>0 and gap>0 and clip>0 and 2*clip<R-r
    holes=[(-(r+R)/2,z) for z in p['back_bolt_z']]
    holes += [(p['crown_bolt_radius']*math.cos(math.radians(a)),
               p['crown_bolt_radius']*math.sin(math.radians(a))) for a in p['crown_bolt_angles_deg']]
    angle=np.linspace(np.pi,0,161)
    boundary=np.array([(-R,-back+clip)]+list(zip(R*np.cos(angle),R*np.sin(angle)))+
        [(R,-lip+clip),(R-clip,-lip),(r+clip,-lip),(r,-lip+clip)]+
        list(zip(r*np.cos(angle[::-1]),r*np.sin(angle[::-1])))+
        [(-r,-back+clip),(-r-clip,-back),(-R+clip,-back)])
    parts=[]
    def add(name,mesh,color,density,role,explode=(0,0,0)):
        assert mesh.body_count==1
        parts.append(dict(name=name,mesh=mesh,color=color,density=density,role=role,explode=np.array(explode,dtype=float)))
    plate_color=[.055,.067,.082]
    for name,y,sign in [('side_plate_front',-(gap+t)/2,-1),('side_plate_rear',(gap+t)/2,1)]:
        add(name,extrude([boundary]+[circle(x,z,hole) for x,z in holes],t,y),
            plate_color,p['assumed_plate_density'],'primary load plate',(0,sign*32,0))
    a,b=np.deg2rad([p['crown_start_deg'],p['crown_end_deg']])
    add('crown_carrier',extrude([sector(p['carrier_inner_radius'],R,a,b)]+
        [circle(x,z,hole) for x,z in holes[2:]],gap),[.11,.13,.15],p['assumed_carrier_density'],
        'cross-width crown support; bolted between plates',(0,0,22))
    add('replaceable_liner',extrude([sector(p['liner_inner_radius'],p['liner_outer_radius'],a,b)],gap),
        [.27,.29,.31],p['assumed_liner_density'],'replaceable contact pad; retention details unresolved',(0,0,-16))
    for i,(x,z) in enumerate(holes[:2]):
        add(f'back_spacer_{i+1}',extrude([circle(x,z,p['spacer_outer_diameter']/2),circle(x,z,hole)],gap),
            [.14,.16,.18],p['assumed_carrier_density'],'spacing sleeve')
    full_width=gap+2*t
    for i,(x,z) in enumerate(holes):
        add(f'fastener_{i+1}_shank',extrude([circle(x,z,p['bolt_shank_diameter']/2)],full_width),
            [.15,.17,.20],7850.,'smooth unthreaded fastener proxy')
        for side,sign in [('head',-1),('nut',1)]:
            yy=sign*(full_width+p['bolt_head_height'])/2
            add(f'fastener_{i+1}_{side}',extrude([circle(x,z,p['bolt_head_diameter']/2,6)],p['bolt_head_height'],yy),
                [.19,.21,.24],7850.,'solid hex proxy; socket/thread details omitted',(0,sign*52,0))
    return parts,holes,boundary


def draw(ax,parts,exploded=False,bar=False):
    light=LightSource(azdeg=230,altdeg=40)
    for part in parts:
        v=part['mesh'].vertices+(part['explode'] if exploded else 0)
        display_color=np.minimum(.14+.8*np.array(part['color']),1)
        ax.add_collection3d(Poly3DCollection(v[part['mesh'].faces],facecolors=display_color,
            linewidths=0,shade=True,lightsource=light))
    if bar:
        aa,yy=np.meshgrid(np.linspace(0,2*np.pi,72),[-52,52])
        ax.plot_surface(15*np.cos(aa),yy,7+15*np.sin(aa),color='#c29e65',alpha=.6,shade=True)
    ax.view_init(elev=22,azim=-58)
    ax.set(xlim=(-52,52),ylim=(-75,75) if exploded else (-52,52),zlim=(-64,64))
    ax.set_box_aspect((104,150 if exploded else 104,128));ax.set_axis_off()


def preview(p,parts,out,mass):
    plt.rcParams['font.family']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig=plt.figure(figsize=(16,10),facecolor='#f1f3f5')
    fig.text(.055,.925,'G1 / MODULAR HOOK',fontsize=29,color='#151b24',weight='bold')
    fig.text(.055,.875,'黑色组合式钩爪 · V2',fontsize=21,color='#263340')
    fig.text(.055,.835,'双侧承力板 / 跨板支撑 / 可换衬块 / 外露紧固件',fontsize=12,color='#61717c')
    ax=fig.add_axes([.00,.23,.52,.59],projection='3d',facecolor='#f1f3f5');draw(ax,parts)
    bx=fig.add_axes([.50,.23,.49,.59],projection='3d',facecolor='#f1f3f5');draw(bx,parts,exploded=True)
    fig.text(.17,.245,'装配状态',fontsize=15,color='#253443')
    fig.text(.69,.245,'结构分解',fontsize=15,color='#253443')
    fig.text(.055,.172,'01  黑色双侧钩板',fontsize=13,color='#152331')
    fig.text(.30,.172,'02  顶部连接支撑件',fontsize=13,color='#152331')
    fig.text(.56,.172,'03  可更换承载衬块',fontsize=13,color='#152331')
    fig.text(.81,.172,'04  间隔套 / 紧固件',fontsize=13,color='#152331')
    fig.text(.055,.111,f'横杆 Ø30 mm     有效承载内径 Ø44 mm     单板厚 4 mm     钩体宽 26 mm     几何质量约 {mass*1000:.0f} g',fontsize=12,color='#4d606d')
    fig.text(.055,.064,'组合结构概念：材料、衬块固定方式与 G1 腕部接口待定；未验证强度或动态承载。',fontsize=11,color='#7e5d3c')
    fig.savefig(out/'preview.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)


def fit_xml(p,parts,out):
    root=ET.Element('mujoco',model='G1 black modular hook V2 - static geometry fixture')
    ET.SubElement(root,'compiler',angle='radian',meshdir='parts')
    assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='.1 -.2 .3',dir='-.2 .4 -1')
    ET.SubElement(world,'geom',name='test_bar',type='capsule',fromto='0 -.08 0 0 .08 0',size=str(p['bar_diameter']/2000),rgba='.7 .55 .3 1')
    body=ET.SubElement(world,'body',name='hook',mocap='true')
    for part in parts:
        name=part['name'];ET.SubElement(assets,'mesh',name=name,file=name+'.stl',scale='.001 .001 .001')
        ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',group='2',
            rgba=' '.join(map(str,part['color']+[1])))
    def arc(name,ri,ro,a,b,y,thickness,count):
        for i in range(count):
            aa=a+(b-a)*i/count;bb=a+(b-a)*(i+1)/count
            points=[(ro*np.cos(aa),ro*np.sin(aa)),(ro*np.cos(bb),ro*np.sin(bb)),
                    (ri*np.cos(bb),ri*np.sin(bb)),(ri*np.cos(aa),ri*np.sin(aa))]
            vertices=[(x/1000,yy/1000,z/1000) for yy in (y-thickness/2,y+thickness/2) for x,z in points]
            nm=f'{name}_{i}'
            ET.SubElement(assets,'mesh',name=nm,vertex=' '.join(str(v) for point in vertices for v in point))
            ET.SubElement(body,'geom',name=nm,type='mesh',mesh=nm,group='3',rgba='.3 .6 .9 .2')
    ri=p['plate_inner_radius'];ro=p['plate_outer_radius'];gap=p['plate_inner_gap'];th=p['plate_thickness']
    for side,y in enumerate((-(gap+th)/2,(gap+th)/2)):
        arc(f'plate_{side}',ri,ro,0,np.pi,y,th,32)
        for name,x,depth in [('back',-(ri+ro)/2,p['back_length']),('lip',(ri+ro)/2,p['lip_length'])]:
            ET.SubElement(body,'geom',name=f'{name}_{side}',type='box',group='3',rgba='.3 .6 .9 .2',
                pos=f'{x/1000} {y/1000} {-depth/2000}',size=f'{(ro-ri)/2000} {th/2000} {depth/2000}')
    a,b=np.deg2rad([p['crown_start_deg'],p['crown_end_deg']])
    arc('liner',p['liner_inner_radius'],p['liner_outer_radius'],a,b,0,gap,24)
    arc('carrier',p['carrier_inner_radius'],p['carrier_outer_radius'],a,b,0,gap,24)
    # Spacers and smooth shanks are represented by solid cylinders (holes omitted).
    for part in parts:
        if 'spacer' not in part['name'] and 'shank' not in part['name']:continue
        bounds=part['mesh'].bounds;c=bounds.mean(axis=0);radius=(bounds[1,0]-bounds[0,0])/2000
        ET.SubElement(body,'geom',name='collision_'+part['name'],type='capsule',group='3',rgba='.3 .6 .9 .2',
            size=str(radius),fromto=' '.join(map(str,[c[0]/1000,bounds[0,1]/1000,c[2]/1000,c[0]/1000,bounds[1,1]/1000,c[2]/1000])))
    # Both bodies are kinematic here. Explicit pairs allow a geometric check
    # without introducing a hidden dynamic constraint or a hanging claim.
    contact=ET.SubElement(root,'contact')
    for geom in body.findall('geom'):
        if geom.get('group')=='3':
            ET.SubElement(contact,'pair',geom1='test_bar',geom2=geom.get('name'))
    ET.indent(root);ET.ElementTree(root).write(out/'hook_fit.xml',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=HERE/'model_final')
    parser.add_argument('--params',type=Path,default=HERE/'parameters.json');args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Choose a new output directory')
    p=json.loads(args.params.read_text(encoding='utf-8'));parts,holes,boundary=create(p)
    out=args.out;(out/'parts').mkdir(parents=True)
    scene=trimesh.Scene();rows=[]
    for part in parts:
        name=part['name'];mesh=part['mesh'];mesh.export(out/'parts'/(name+'.stl'))
        check=trimesh.load_mesh(out/'parts'/(name+'.stl'),process=True)
        assert check.is_watertight and check.is_winding_consistent
        mass=mesh.volume*1e-9*part['density']
        rows.append(dict(name=name,role=part['role'],volume_mm3=float(mesh.volume),estimated_mass_kg=float(mass),
            assumed_density_kg_m3=part['density'],watertight=True,color_rgb=part['color']))
        glb=mesh.copy();glb.apply_scale(.001)
        glb.visual=trimesh.visual.TextureVisuals(material=PBRMaterial(name=name,
            baseColorFactor=[round(c*255) for c in part['color']]+[255],metallicFactor=.65 if 'liner' not in name else 0.,roughnessFactor=.38 if 'liner' not in name else .72))
        scene.add_geometry(glb,node_name=name,geom_name=name)
    (out/'assembly.glb').write_bytes(scene.export(file_type='glb'))
    reload=trimesh.load_scene(out/'assembly.glb');assert len(reload.geometry)==len(parts)
    assembly=trimesh.util.concatenate([part['mesh'] for part in parts]);assembly.export(out/'assembly_mm.stl')
    total_mass=sum(row['estimated_mass_kg'] for row in rows)
    preview(p,parts,out,total_mass);fit_xml(p,parts,out)
    (out/'parameters.json').write_bytes(args.params.read_bytes());(out/'build_model.py').write_bytes(Path(__file__).read_bytes())
    manifest=dict(revision=p['revision'],part_count=len(parts),all_parts_watertight=True,glb_roundtrip_part_count=len(reload.geometry),
        estimated_total_mass_kg=total_mass,bar_diameter_mm=p['bar_diameter'],
        plate_spacing_mm=p['plate_inner_gap'],body_width_mm=p['plate_inner_gap']+2*p['plate_thickness'],
        width_including_fasteners_mm=p['plate_inner_gap']+2*p['plate_thickness']+2*p['bolt_head_height'],
        effective_seat_diameter_mm=2*p['liner_inner_radius'],parts=rows,
        scope='Black modular concept; no verified material choice, liner retention, wrist fit, strength or dynamic hanging',
        collision_scope='Static mocap geometry only; conservative boxes/capsules omit corner clips, holes and threads',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files_sha256={str(f.relative_to(out)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('part_count','all_parts_watertight','estimated_total_mass_kg','body_width_mm','width_including_fasteners_mm')},indent=2))


if __name__=='__main__':main()
