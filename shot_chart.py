import math
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Rectangle
from nba_api.stats.endpoints.shotchartdetail import ShotChartDetail
from nba_api.stats.static import teams


# https://github.com/bradleyfay/py-Goldsberry/blob/master/docs/Visualizing%20NBA%20Shots%20with%20py-Goldsberry.ipynb
def draw_court(
    ax=None,
    color='black',
    lw=2,
    outer_lines=False,
    center_court=False,
    inner_box=False,
    box_dashes=True,
    lower_free_throw=True,
    dashed_free_throw=True,
):
    # 1 unit is equal to 0.1'
    # dimensions from https://official.nba.com/rule-no-1-court-dimensions-equipment/

    # get current axis if one is not provided
    if ax is None:
        ax = plt.gca()

    # patch arguments
    args = {'linewidth': lw, 'color': color, 'fill': False}

    # hoop diameter is 18", radius of 9" (0.75')
    radius = 7.5
    hoop = Circle((0, 0), radius=radius, **args)

    # backboard is 15" from center of basket (6" from closest part of hoop)
    # backboard is 6' wide centered around the basket
    offset = 5
    bb_width = 60
    y = -radius - offset
    backboard = Rectangle((-bb_width / 2, y), bb_width, -1, **args)
    hoop_connect = Rectangle((-1, y), 2, offset, linewidth=lw, color=color)

    # outer box of the paint (width=16ft, height=19ft)
    bottom = -52.5
    paint_width = 160
    paint_height = 190
    paint_width_2 = paint_width / 2
    outer_box = Rectangle((-paint_width_2, bottom), paint_width, paint_height, **args)

    # top arc of free throw line
    free_throw_center = paint_height + bottom
    paint_width_inner = 120
    top_free_throw = Arc(
        (0, free_throw_center),
        paint_width_inner,
        paint_width_inner,
        theta1=0,
        theta2=180,
        **args,
    )

    # restricted zone (arc with 4ft radius from center of the hoop)
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, **args)

    # corner 3pt lines
    # measured as the distance from basket to baseline (4'9") plus basket to 3pt break
    # some dimensions quote 14', but this is slightly high
    # official rules do not provide a length for the corner 3pt lines
    # rules merely state the lines intersect the arc
    r_arc = 237.5
    d_corner = 220
    length = math.sqrt(r_arc ** 2 - d_corner ** 2) - bottom
    corner_three_left = Rectangle((-d_corner, bottom), 0, length, **args)
    corner_three_right = Rectangle((d_corner, bottom), 0, length, **args)

    # 3pt arc
    # radius of 23'9" from center of hoop and intersets corner 3pt lines
    # angle between lines from basket to corner three and from basket to 3pt break
    theta = math.acos(d_corner / r_arc) * 180 / math.pi
    size = 2 * r_arc
    three_arc = Arc(
        (0, 0), size, size, theta1=theta, theta2=180 - theta, capstyle='round', **args
    )

    # list of court elements
    court_elements = [
        hoop,
        backboard,
        hoop_connect,
        outer_box,
        top_free_throw,
        restricted,
        corner_three_left,
        corner_three_right,
        three_arc,
    ]

    # dashes on sides of the paint
    if box_dashes:
        dash_width = 5
        dash_left = [-paint_width_2 - dash_width, paint_width_2]
        dash_bottom = np.array([0, 10, 40, 70]) + 70 + bottom
        positions = product(dash_left, dash_bottom)
        lines = [Rectangle((x, y), dash_width, 0, **args) for x, y in positions]
        court_elements.extend(lines)

    # bottom arc of free throw line
    if lower_free_throw:
        num_segments = 8 if dashed_free_throw else 1
        start_theta = 180
        theta = 180 / (2 * num_segments - 1)

        for i in range(num_segments):
            arc = Arc(
                (0, free_throw_center),
                paint_width_inner,
                paint_width_inner,
                theta1=start_theta,
                theta2=start_theta + theta,
                **args,
            )
            court_elements.append(arc)
            start_theta += 2 * theta

    # inner box of the paint (width=12ft, height=19ft)
    if inner_box:
        inner_box = Rectangle(
            (-paint_width_inner / 2, bottom), paint_width_inner, paint_height, **args
        )
        court_elements.append(inner_box)

    half_court_height = 470
    court_width = 500
    bottom_left = -court_width / 2, bottom

    # center court circles
    if center_court:
        center = 0, half_court_height + bottom
        center_arcs = [
            Arc(center, 40, 40, theta1=180, theta2=0, **args),
            Arc(center, 120, 120, theta1=180, theta2=0, **args),
        ]
        court_elements.extend(center_arcs)

    # court boundaries (half court line, baseline, and side out lines)
    if outer_lines:
        outer_lines = Rectangle(bottom_left, court_width, half_court_height, **args)
        court_elements.append(outer_lines)
    else:
        base_line = Rectangle(bottom_left, court_width, 0, **args)
        court_elements.append(base_line)

    # add court elements
    for element in court_elements:
        ax.add_patch(element)

    return ax


def get_league_shots(season):
    df = None

    for team in teams.get_teams():
        data = ShotChartDetail(
            team_id=team['id'],
            player_id=0,
            context_measure_simple='FGA',
            season_nullable=season,
        )

        if df is None:
            df = data.get_data_frames()[0]
        else:
            df = df.append(data.get_data_frames()[0])

    return df


def shot_chart(
    shots_df,
    ax,
    threshold=None,
    topn=None,
    color='pct',
    markersize=[50, 200],
    nbins=4,
    gridsize=50,
    title=None,
    scatter_args={},
    legend_args={},
    title_args={},
):
    fig = plt.figure()
    extent = (-250, 250, -52.5, 417.5)
    shots_hex = plt.hexbin(
        shots_df.LOC_X, shots_df.LOC_Y, gridsize=gridsize, extent=extent, mincnt=1
    )
    shots_array = shots_hex.get_array()

    if threshold is not None:
        mask = shots_array >= threshold
    elif topn is not None:
        mask = shots_array >= sorted(shots_array)[-topn]
    else:
        mask = np.ones_like(shots_array, dtype=bool)

    if color == 'pct':
        pct_hex = plt.hexbin(
            shots_df.LOC_X,
            shots_df.LOC_Y,
            C=shots_df.SHOT_MADE_FLAG,
            gridsize=gridsize,
            extent=extent,
        )
        c = pct_hex.get_array()[mask] * 100
        legend_title = 'FG%'
    elif color == 'efg':
        flag = shots_df.SHOT_MADE_FLAG.values.astype(float)
        flag[flag & (shots_df.SHOT_TYPE == '3PT Field Goal')] = 1.5
        pct_hex = plt.hexbin(
            shots_df.LOC_X, shots_df.LOC_Y, C=flag, gridsize=gridsize, extent=extent
        )
        c = pct_hex.get_array()[mask] * 100
        legend_title = 'eFG%'
    elif color == 'pts':
        pts = shots_df.apply(
            lambda row: int(row.SHOT_TYPE[0]) * row.SHOT_MADE_FLAG, axis=1
        )
        pts_hex = plt.hexbin(
            shots_df.LOC_X, shots_df.LOC_Y, C=pts, gridsize=gridsize, extent=extent
        )
        c = pts_hex.get_array()[mask]
        legend_title = 'Points per Attempt'
    else:
        c = color

    if type(markersize) in {list, tuple}:
        bins, breaks = pd.qcut(
            shots_array[mask], nbins, labels=False, retbins=True, duplicates='drop'
        )
        binsizes = np.linspace(markersize[0], markersize[1], len(breaks) - 1)
        s = [binsizes[b] for b in bins]
    else:
        s = markersize

    plt.close(fig)

    offsets = shots_hex.get_offsets()
    x = offsets[mask, 0]
    y = offsets[mask, 1]

    _scatter_args = {'cmap': 'coolwarm'}
    _scatter_args.update(scatter_args)

    scatter = ax.scatter(x, y, c=c, s=s, marker='h', **_scatter_args)

    _legend_args = {
        'fontsize': 14,
        'title_fontsize': 14,
        'columnspacing': 0.25,
        'handletextpad': 0,
        'frameon': False,
        'bbox_transform': ax.transData,
    }
    _legend_args.update(legend_args)

    if color in {'pct', 'efg', 'pts'}:
        handles_color, labels = scatter.legend_elements(num=5)
        handles_size, _ = scatter.legend_elements(prop='sizes')
        handles = [Line2D([0], [0]) for _ in handles_color]

        for h, hc in zip(handles, handles_color):
            h.update_from(handles_size[-1])
            h.set_color(hc.get_color())

        legend = ax.legend(
            handles,
            labels,
            loc='upper left',
            title=legend_title,
            bbox_to_anchor=(-260, -50),
            ncol=len(handles),
            **_legend_args,
        )
        ax.add_artist(legend)

    if type(markersize) in {list, tuple}:
        handles, _ = scatter.legend_elements(prop='sizes', alpha=0.6, markeredgewidth=0)
        labels = np.linspace(0, 100, len(breaks), dtype=int)
        ax.legend(
            handles,
            labels[1:],
            loc='upper right',
            title='Attempts Percentile',
            bbox_to_anchor=(260, -50),
            ncol=len(handles),
            **_legend_args,
        )

    if title is not None:
        _title_args = {'fontsize': 20}
        _title_args.update(title_args)
        ax.set_title(title, **_title_args)
