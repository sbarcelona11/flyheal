"""Replay saved 2D foraging policies; optionally export a comparative GIF."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from .ui import ReplayStore


def draw_world(world):
    image = Image.new('RGB', (480, 320), '#111f20')
    draw = ImageDraw.Draw(image)
    scale = 28
    def xy(x, y):
        return (100 + x * scale, 300 - y * scale)
    for ob in world['obstacles']:
        x, y = xy(ob['x'], ob['y'] + ob['height'])
        draw.rectangle((x, y, x + ob['width'] * scale, y + ob['height'] * scale), fill='#657980')
    for food in world['foods']:
        x, y = xy(food['x'], food['y'])
        draw.ellipse((x-7, y-7, x+7, y+7), fill='#52664b' if food['collected'] else '#b4e584')
    import math
    fly = world['fly']; x, y = xy(fly['x'], fly['y'])
    draw.ellipse((x-4, y-4, x+4, y+4), fill='#d9e3c7')
    draw.line((x, y, x+12*math.cos(fly['heading']), y-12*math.sin(fly['heading'])), fill='#8fc9be', width=3)
    return image


def play(results, architecture='structured', seed=0, condition='neurons_0.5',
         stage='all', episodes=1, headless=False, gif=None):
    if episodes != 1:
        raise ValueError('La demo reproduce un episodio reservado; usá experiment para evaluar varios')
    store = ReplayStore(results)
    stages = ['baseline', 'post', 'recovered'] if stage == 'all' else [stage]
    replays = {phase: store.replay(architecture, seed, condition, phase) for phase in stages}
    returns = {phase: [replay['return']] for phase, replay in replays.items()}
    if gif:
        clips = {phase: [draw_world(frame['world']) for frame in replay['frames'][::4]] for phase, replay in replays.items()}
        frames = []
        for index in range(max(map(len, clips.values()))):
            image = Image.new('RGB', (480 * len(stages), 370), 'white')
            draw = ImageDraw.Draw(image)
            for column, phase in enumerate(stages):
                image.paste(clips[phase][min(index, len(clips[phase])-1)], (480*column, 50))
                draw.text((480*column+12, 10), f'{architecture} | {phase} | retorno {returns[phase][0]:.2f}', fill='black')
                draw.text((480*column+12, 28), replays[phase]['provenance'], fill='black')
            frames.append(image)
        gif = Path(gif); gif.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=80, loop=0)
    for phase, values in returns.items():
        print(f'{architecture} semilla={seed} {phase}: retorno={values[0]:.2f}')
    if not headless and not gif:
        print('Para ver la simulación interactiva: python -m flyheal.ui --results ' + str(results))
    return returns


def main():
    parser = argparse.ArgumentParser(description='Reproducir políticas en el mundo 2D')
    parser.add_argument('--results', type=Path, default=Path('results/new_run'))
    parser.add_argument('--architecture', choices=['dense','structured','random','male_cns','rewired_male_cns'], default='male_cns')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--condition', default='neurons_0.5')
    parser.add_argument('--stage', choices=['all','baseline','post','recovered'], default='all')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--gif', type=Path)
    play(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
