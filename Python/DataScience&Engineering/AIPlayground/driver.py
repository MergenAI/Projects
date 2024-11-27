import pygame
from TetrisGame import Tetris
from TetrisRL import Agent, DQN
import numpy as np

clock = pygame.time.Clock()


# smarter state function
def get_state(game):
    # Board matrix: binary matrix with 1 for filled cells, 0 for empty cells
    board_state = np.array(game.field, dtype=float).flatten()

    # Current piece position and orientation
    if game.figure is not None:
        current_piece_type = game.figure.type  # Assuming each piece has a type identifier
        current_piece_orientation = game.figure.rotation
        current_piece_x = game.figure.x
        current_piece_y = game.figure.y
    else:
        current_piece_type = -1
        current_piece_orientation = -1
        current_piece_x = -1
        current_piece_y = -1

    # Next piece information
    # if game.next_figure is not None:
    #     next_piece_type = game.next_figure.type  # Assuming each piece has a type identifier
    # else:
    #     next_piece_type = -1

    # Concatenate all features into a single state vector
    state = np.concatenate([
        board_state,
        [current_piece_type, current_piece_orientation, current_piece_x, current_piece_y],
        # [next_piece_type]
    ])

    return state


# Convert the game field to a suitable state format for the agent
# def get_state(game):
#     # Convert game.field to a binary matrix, e.g., 1 for filled tiles and 0 for empty tiles
#     state = np.array(game.field, dtype=float)
#     return state


def custom_reward_1(game, lines_cleared, reward,episodes_survived):
    state = np.array(game.field, dtype=float)

    max_height = max(np.nonzero(state)[0]) if np.any(state) else 0
    height_penalty = -max_height * 2  # Adjust scale as needed to lower its impact

    holes = 0
    for col in range(state.shape[1]):
        column = state[:max_height + 1, col]
        filled = np.where(column > 0)[0]
        if len(filled) > 0:
            holes += np.sum(column[:filled[0]] == 0)
    hole_penalty = -holes  # Mild penalty per hole

    # Line clear reward with exponential scaling
    line_clear_reward = 10 * (2 ** lines_cleared)  # Adjust scale for greater line-clearing impact

    # Survival reward with cumulative scaling
    survival_reward = 3 * episodes_survived

    # Total reward calculation
    total_reward = reward + height_penalty + hole_penalty + line_clear_reward + survival_reward
    return total_reward


# Action mapping
action_map = {
    0: lambda game: game.go_side(-1),  # Left
    1: lambda game: game.go_side(1),  # Right
    2: lambda game: game.rotate(),  # Rotate
    3: lambda game: game.go_down(),  # Down
    4: lambda game: game.go_space(),  # Drop
}
pygame.init()

# Initialize the game and agent
game = Tetris(20, 10)

pressing_down = False
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

size = (400, 500)
screen = pygame.display.set_mode(size)

colors = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
]
# Parameters
# Parameters
batch_size = 256
replay_interval = 32
episode = 0
episodes_survived= 0
best_survived=0
best_score=0
done = False
fps = 25
game = Tetris(20, 10)

# Initialize the game and agent
agent = Agent(action_size=len(action_map), max_mem_size=1000, state_size=get_state(game).shape)
reward = 0
while True:
    if game.figure is None:
        game.new_figure()
    # Reset the game if the previous episode is done
    if done:
        game = Tetris(20, 10)
        done = False
        episode += 1  # Increment episode only once per completed game

    # Generate the current state
    state = get_state(game)

    # Agent selects an action
    action = agent.act(state)

    # Execute the selected action in the game
    # action_map[action](game)
    if action == 0:
        game.go_side(-1)
    elif action == 1:
        game.go_side(1)
    elif action == 2:
        game.rotate()
    elif action == 3:
        game.go_down()
    elif action == 4:
        game.go_space()

    else:
        assert "unknown action"
    # Define reward structure
    if game.state == "gameover":
        best_survived=episodes_survived if best_survived<episodes_survived else best_survived
        best_score=game.score if best_score<game.score else best_score
        print("best score: ", best_score, "best survived ", best_survived)
        next_state = get_state(game)
        agent.remember(state, action, reward-1000, next_state, done)
        episodes_survived=0
        reward = 0  # Penalty for game over

        done = True
    else:
        # reward = game.score  # You may enhance this with rewards for clearing lines, etc.
        if game.placed:
            reward = custom_reward_1(game, game.score, reward,episodes_survived)
            game.placed = False
    # Get the next state after performing the action
    next_state = get_state(game)
    print(f"Episode {episode}, Action: {action}, Reward: {reward}")

    # Rendering and game display updates
    screen.fill(WHITE)
    for i in range(game.height):
        for j in range(game.width):
            pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom], 1)
            if game.field[i][j] > 0:
                pygame.draw.rect(screen, colors[game.field[i][j]],
                                 [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 2, game.zoom - 1])

    if game.figure is not None:
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in game.figure.image():
                    pygame.draw.rect(screen, colors[game.figure.color], [game.x + game.zoom * (j + game.figure.x) + 1,
                                                                         game.y + game.zoom * (i + game.figure.y) + 1,
                                                                         game.zoom - 2, game.zoom - 2])

    font = pygame.font.SysFont('Calibri', 25, True, False)
    text = font.render("Score: " + str(game.score), True, BLACK)
    text1 = font.render("Best Score: " + str(best_score), True, BLACK)
    text2 = font.render("Best Survived: " + str(best_survived), True, BLACK)
    screen.blit(text, [0, 0])
    screen.blit(text1, [0, 20])
    screen.blit(text2, [0, 30])

    if game.state == "gameover":
        font1 = pygame.font.SysFont('Calibri', 65, True, False)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))
        screen.blit(text_game_over, [20, 200])
        screen.blit(text_game_over1, [25, 265])

    pygame.display.flip()
    clock.tick(fps)

    # Save experience to replay memory
    agent.remember(state, action, reward, next_state, done)

    # Train the agent after every episode or at specified intervals
    if len(agent.memory) >= batch_size and episode % replay_interval == 0:
        agent.replay(batch_size)

    episode += 1
    episodes_survived+= 1

    """
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Daten vorbereiten
        altersgruppen = ['0-17 Jahre', '18-35 Jahre', '36-55 Jahre', '56+ Jahre']
        maennlich = {
            'Diagnose A': [10, 15, 20, 10],
            'Diagnose B': [5, 10, 15, 8],
            'Diagnose C': [2, 5, 10, 6]
        }
        weiblich = {
            'Diagnose A': [8, 12, 18, 9],
            'Diagnose B': [6, 14, 13, 7],
            'Diagnose C': [3, 7, 5, 5]
        }
        padding = 0.05
        x = np.arange(len(altersgruppen))  # X-Achsen-Positionen
        width = 0.4  # Breite der Balken
        farbe_maennlich = '#0D92F4'  # Blau für Männer
        farbe_weiblich = '#C62E2E'   # Rot für Frauen
        # Balkendiagramm erstellen
        fig, ax = plt.subplots(figsize=(10, 6))
        farben = ['#72BF78', '#A0D683', '#D3EE98']
        linewidth=5
        # Daten für Männlich
        ax.bar(-padding + x - width / 2, maennlich['Diagnose A'], width, label='Diagnose A', color=farben[0], edgecolor=farbe_maennlich,
               linewidth=linewidth)
        ax.bar(-padding + x - width / 2, maennlich['Diagnose B'], width, bottom=maennlich['Diagnose A'], label='Diagnose B',
               color=farben[1], edgecolor=farbe_maennlich, linewidth=linewidth)
        ax.bar(-padding + x - width / 2, maennlich['Diagnose C'], width,
               bottom=np.array(maennlich['Diagnose A']) + np.array(maennlich['Diagnose B']), label='Diagnose C',
               color=farben[2], edgecolor=farbe_maennlich, linewidth=linewidth)
        
        # Daten für Weiblich
        ax.bar(x + width / 2, weiblich['Diagnose A'], width, color=farben[0], edgecolor=farbe_weiblich, linewidth=linewidth)
        ax.bar(x + width / 2, weiblich['Diagnose B'], width, bottom=weiblich['Diagnose A'], color=farben[1], edgecolor=farbe_weiblich,
               linewidth=linewidth)
        ax.bar(x + width / 2, weiblich['Diagnose C'], width,
               bottom=np.array(weiblich['Diagnose A']) + np.array(weiblich['Diagnose B']), color=farben[2], edgecolor=farbe_weiblich,
               linewidth=linewidth)
        
        # Diagrammeinstellungen
        ax.set_xlabel('Altersgruppe')
        ax.set_ylabel('Anzahl der Diagnosen')
        ax.set_title('Verteilung der Diagnosen nach Altersgruppe und Geschlecht')
        ax.set_xticks(x)
        ax.set_xticklabels(altersgruppen)
        handles, labels = ax.get_legend_handles_labels()
        custom_legend = [plt.Line2D([0], [0], color=farbe_maennlich, lw=4, label='Männlich'),
                         plt.Line2D([0], [0], color=farbe_weiblich, lw=4, label='Weiblich'),
                         plt.scatter([0], [0], color=farben[0], lw=4, label='Diagnose A'),
                         plt.scatter([0], [0], color=farben[1], lw=4, label='Diagnose B'),
                         plt.scatter([0], [0], color=farben[2], lw=4, label='Diagnose C')]
        ax.legend(handles=custom_legend, title="Geschlecht", loc="upper right")
        ax.set_facecolor('#EEEDED')
        plt.tight_layout()
        plt.show()
    """
