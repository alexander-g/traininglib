import typing as tp
from xml.dom import minidom


Point     = tp.Tuple[float, float]  # x, y
Component = tp.List[Point]
Path      = tp.List[Component]
Size      = tp.Tuple[int, int]      # width, height

class ParsedSVG(tp.NamedTuple):
    paths: tp.List[Path]
    size:  Size

def parse_svg(svgpath:str) -> ParsedSVG:
    '''Parse a GIMP path svg file'''
    svgtext  = open(svgpath).read()
    return parse_svg_string(svgtext)

def parse_point_str(p:str) -> Point:
    '''Parse a svg point string "x.xxx,y.yyy" to a float tuple.'''
    x,y = p.split(',')
    return float(x), float(y)

def parse_svg_path_d(d:str) -> Path:
    '''Parse the `d` attribute of a <path>. (limited capabilities)'''
    paths        = []
    segments_str = d.split('C ')[1:]
    for segment_str in segments_str:
        segment = []
        lines = segment_str.split('             ')
        for i, line in enumerate(lines):
            line = line.strip()
            line = line.split('M')[0]
            if len(line) == 0:
                continue
            points_str = line.split(' ')
            if i == 0:
                segment.append(
                    parse_point_str(points_str[0])
                )
            segment.append(
                parse_point_str(points_str[-1])
            )
        paths.append(segment)
    return paths

def parse_size_from_viewbox(viewbox_str:str) -> Size:
    W,H = viewbox_str.split(' ')[-2:]
    return int(W), int(H)

def parse_svg_string(svgstring:str) -> ParsedSVG:
    svgdom   = minidom.parseString(svgstring)
    svg_el   = svgdom.getElementsByTagName('svg')
    assert len(svg_el) == 1
    size     = parse_size_from_viewbox(
        svg_el[0].getAttribute('viewBox')
    )

    svgpaths = svgdom.getElementsByTagName("path")
    paths    = []
    for path in svgpaths:
        svg_d  = path.getAttribute('d')
        parsed = parse_svg_path_d(svg_d)
        paths.append(parsed)
    return ParsedSVG(paths, size)


def export_paths_as_svg(paths:tp.List[Path], size:Size) -> str:
    '''Export a list of paths to the same SVG format as GIMP'''
    impl = minidom.getDOMImplementation()
    assert impl is not None
    doc = impl.createDocument(None, "svg", None)
    svg = doc.documentElement

    W,H = size
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
    svg.setAttribute("width", f"{W}px") 
    svg.setAttribute("height", f"{H}px")
    svg.setAttribute("viewBox", f"0 0 {W} {H}")

    for i, path in enumerate(paths):

        svgpath = doc.createElement("path")
        svgpath.setAttribute("id", f"Path #{i}")
        svgpath.setAttribute("fill", "none")
        svgpath.setAttribute("stroke", "black")
        svgpath.setAttribute("stroke-width", "1")

        d_string  = ''
        for j, component in enumerate(path):
            if j == 0:
                p0 = x,y = component[0]
                d_string += f'             M {x},{y}\n'

            for k in range(len(component)-1):
                p0 = x0,y0 = component[k]
                p1 = x1,y1 = component[k+1]
                
                d_string += '             '
                if k == 0:
                    d_string += 'C '
                d_string += f'{x0},{y0} {x1},{y1} {x1},{y1}\n'

        svgpath.setAttribute("d", d_string)
        svg.appendChild(svgpath)

    return doc.toprettyxml()
