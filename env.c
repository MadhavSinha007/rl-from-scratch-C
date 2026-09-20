#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(ms) Sleep(ms)
#else
#include <unistd.h>
#define sleep_ms(ms) usleep((ms) * 1000)
#endif

#include "base.h"
#include "arena.h"
#include "prng.h"
#include "mat.h"
#include "autograd.h"
#include "model.c"

typedef enum {
    LEFT = 0,
    RIGHT = 1,
    UP = 2,
    DOWN = 3,
    NONE = 4
} ACTION;

typedef struct {
    i32 x;
    i32 y;
} State;

typedef struct {
    State snake;      // head position (mirrors body[0])
    State* body;      // body[0] = head, body[1..length-1] = tail segments
    u32 length;        // number of segments currently alive, including head
    b32 grow_pending;  // set when food was eaten; applied on the next move
    State food;
    f32 score;
    u32 foods_eaten;
    u32 steps_since_food; // steps taken since the last meal (starvation clock)
    u32 rows;
    u32 cols;
    u32 grid_size;
    u64 steps;
    ACTION pov;
} SnakeENV;

// --- Reward shaping constants -------------------------------------------
#define LIVING_PENALTY       -0.01f
#define DISTANCE_REWARD        0.2f  // toward/away from food, per step
#define FOOD_REWARD            50.0f
#define DEATH_PENALTY         -10.0f
#define STARVE_PENALTY_SCALE  -0.002f // extra penalty per step, scaled by hunger
#define STARVE_LIMIT           50     // steps without food before starving to death

#define BUFFER_SIZE 64
#define EPISODE_LEN 100

typedef struct {
    State states[EPISODE_LEN];
    State food_states[EPISODE_LEN];
    ACTION povs[EPISODE_LEN];
    ACTION actions[EPISODE_LEN];
    f32 rewards[EPISODE_LEN];
    f32 returns[EPISODE_LEN];
    u32 len;
} Trajectory;

typedef struct {
    Trajectory trajectories[BUFFER_SIZE];
    u32 count;
} ReplayBuffer;

// Initialize snake environment instance
SnakeENV* create_env(u32 grid_size) {
    SnakeENV* env = malloc(sizeof(*env));
    if (!env) return NULL;

    u32 side = (u32)sqrtf((f32)grid_size);
    if (side * side != grid_size) {
        free(env);
        return NULL;
    }

    env->rows = side;
    env->cols = side;
    env->grid_size = grid_size;

    // A snake can never be longer than the number of cells on the board
    env->body = malloc(sizeof(State) * grid_size);
    if (!env->body) {
        free(env);
        return NULL;
    }
    env->length = 0;

    return env;
}

// Free everything allocated by create_env
void destroy_env(SnakeENV* env) {
    if (!env) return;
    free(env->body);
    free(env);
}

b32 position_on_snake(SnakeENV* env, State pos);

// Uniform random integer in [0, n)
static i32 uniform_randi(u32 n) {
    i32 r = (i32)(prng_randf() * (f32)n);
    if (r < 0) r = 0;
    if (r >= (i32)n) r = (i32)n - 1;
    return r;
}

// Generate uniform random position inside grid, optionally avoiding the snake
State get_random_food_loc(SnakeENV* env) {
    State loc;
    do {
        loc = (State) {
            .x = uniform_randi(env->cols),
            .y = uniform_randi(env->rows),
        };
    } while (position_on_snake(env, loc));
    return loc;
}

// Check whether a cell is occupied by any part of the snake's body
b32 position_on_snake(SnakeENV* env, State pos) {
    for (u32 i = 0; i < env->length; i++) {
        if (env->body[i].x == pos.x && env->body[i].y == pos.y) {
            return true;
        }
    }
    return false;
}

// Check boundary, self-collision, and starvation
b32 game_over(SnakeENV* env) {
    // Wall collision
    if (env->snake.x < 0 || env->snake.x >= (i32)env->cols ||
        env->snake.y < 0 || env->snake.y >= (i32)env->rows) {
        return true;
    }
    // Self collision: head hitting any trailing body segment
    for (u32 i = 1; i < env->length; i++) {
        if (env->body[i].x == env->snake.x && env->body[i].y == env->snake.y) {
            return true;
        }
    }
    // Starvation: too long without eating forces a death, so a policy that's
    // scared to approach food can't just loop safely forever.
    if (env->steps_since_food >= STARVE_LIMIT) {
        return true;
    }
    return false;
}

// Calculate Manhattan distance between two points
i32 get_distance(State a, State b) {
    return abs(a.x - b.x) + abs(a.y - b.y);
}

// Calculate reward, queue growth, and respawn food if eaten
f32 get_reward(SnakeENV* env, State old_snake) {
    env->steps_since_food++;

    f32 reward = LIVING_PENALTY + STARVE_PENALTY_SCALE * (f32)env->steps_since_food;

    // REWARD SHAPING: Breadcrumbs to stop circle-running
    i32 old_dist = get_distance(old_snake, env->food);
    i32 new_dist = get_distance(env->snake, env->food);

    if (new_dist < old_dist) {
        reward += DISTANCE_REWARD;  // Moving towards food!
    } else {
        reward -= DISTANCE_REWARD;  // Moving away from food!
    }

    if (env->snake.x == env->food.x && env->snake.y == env->food.y) {
        reward += FOOD_REWARD;
        env->foods_eaten++;
        env->steps_since_food = 0;

        // Defer the actual growth to the next move (see take_action). Growing
        // right here would duplicate a segment onto the head's own current
        // cell -- for a length-1 snake that duplicate IS the head, which
        // game_over() would then read as an instant, bogus self-collision
        // (this is exactly the bug that crept back into this version -- the
        // first food eaten each episode was killing the snake on the spot).
        if (env->length < env->grid_size) {
            env->grow_pending = true;
        }

        env->food = get_random_food_loc(env);
    }

    if (game_over(env)) {
        reward += DEATH_PENALTY; // covers wall, self-collision, AND starvation
    }

    return reward;
}

// Reset environment state for a new episode
void reset_state(SnakeENV* env) {
    env->snake = (State){
        .x = (i32)(env->cols / 2),
        .y = (i32)(env->rows / 2),
    };
    env->length = 1;
    env->grow_pending = false;
    env->steps_since_food = 0;
    env->body[0] = env->snake;

    env->food = get_random_food_loc(env); // avoids the whole snake body

    env->score = 0.0f;
    env->foods_eaten = 0;
    env->pov = RIGHT;
    env->steps = 0;
}

// Resolve NONE and the anti-180-reversal rule into a concrete action, the
// same way take_action does internally. Pulled out on its own so anything
// that wants to know "what would actually happen" (e.g. the test-time loop
// breaker below) can ask without duplicating -- and risking drifting from --
// this logic.
ACTION resolve_action(SnakeENV* env, ACTION action) {
    if (action == NONE) {
        action = env->pov;
    }

    // Prevent 180 degree instant reversal (wiggling)
    if ((action == LEFT && env->pov == RIGHT) ||
        (action == RIGHT && env->pov == LEFT) ||
        (action == UP && env->pov == DOWN) ||
        (action == DOWN && env->pov == UP)) {
        action = env->pov;
    }

    return action;
}

// Where would the head end up if `action` were resolved and applied right
// now? Pure/read-only -- does not touch env.
State peek_next_head(SnakeENV* env, ACTION action) {
    action = resolve_action(env, action);

    State next = env->snake;
    if (action == LEFT)       next.x -= 1;
    else if (action == RIGHT) next.x += 1;
    else if (action == UP)    next.y += 1;
    else if (action == DOWN)  next.y -= 1;

    return next;
}

// Update snake coordinates based on selected action
void take_action(SnakeENV* env, ACTION action) {
    action = resolve_action(env, action);

    // Apply any growth queued up by eating food last step. Extending the
    // length BEFORE the shift below means the shift loop also fills the new
    // trailing slot -- with the position the old tail (or old head, if the
    // snake was length 1) occupied a moment ago, not the current head.
    if (env->grow_pending) {
        env->length++;
        env->grow_pending = false;
    }

    // Shift every trailing segment into the position the segment ahead of
    // it currently occupies (classic array-based snake movement). This has
    // to happen BEFORE the head moves, so body[1] ends up where the head
    // used to be.
    for (i32 i = (i32)env->length - 1; i >= 1; i--) {
        env->body[i] = env->body[i - 1];
    }

    if (action == LEFT)       env->snake.x -= 1;
    else if (action == RIGHT) env->snake.x += 1;
    else if (action == UP)    env->snake.y += 1;
    else if (action == DOWN)  env->snake.y -= 1;

    env->body[0] = env->snake;
    env->pov = action;
    env->steps++; // was declared and reset but never actually counted anywhere
}

// Sample discrete action from softmax probability distribution
ACTION sample_action(matrix* probs) {
    f32 r = prng_randf();
    f32 cdf = 0.0f;
    u32 size = probs->rows * probs->cols;

    for (u32 i = 0; i < size; i++) {
        cdf += probs->data[i];
        if (r <= cdf) {
            return (ACTION)i;
        }
    }
    return (ACTION)(size - 1);
}

// Choose greedy action for evaluation / testing
ACTION greedy_action(matrix* probs) {
    u32 size = probs->rows * probs->cols;
    u32 best_idx = 0;
    f32 best_val = -1.0f;
    for (u32 i = 0; i < size; i++) {
        if (probs->data[i] > best_val) {
            best_val = probs->data[i];
            best_idx = i;
        }
    }
    return (ACTION)best_idx;
}

// --- Test-time loop breaker -------------------------------------------
//
// BUG THIS FIXES: a policy trained with REINFORCE is a *distribution*, and
// during training actions are sampled from it (sample_action), so even a
// state where the top action leads nowhere useful is escaped sooner or
// later by chance -- which is exactly why training shows the agent eating
// food constantly. run_test_agent instead calls greedy_action, which is
// perfectly deterministic: for a given state it always returns the same
// action. Combined with the anti-180-reversal rule in take_action (which
// removes the snake's one way to "undo" a step), this means that if the
// argmax choice at some cell points back toward a cell the snake just
// came from, the snake ends up walking a *fixed loop* -- sometimes a tiny
// one right next to the food, sometimes a big lap around most of the
// board -- forever, and dies to the starvation clock having eaten
// nothing. This is exactly the "just moves and starves" symptom:
// reproduce it with `./env play` on a fully-trained model and roughly
// half of episodes end at exactly step 50 (STARVE_LIMIT) with 0 food
// eaten.
//
// Because these loops can be as large as the whole board, trying to
// detect them by remembering "the last N cells visited" needs N to be
// unrealistically large to catch every case. The robust fix is simpler:
// lean on the fact that the *stochastic* policy already reliably finds
// food (that's what training itself demonstrated). Stay greedy normally
// -- it's usually the best single action -- but once the snake has gone
// suspiciously long without eating (well before the hard starvation
// limit), assume it's stuck in a loop and switch to sampling from the
// same trained distribution instead of always taking the top pick. A
// stall threshold well under STARVE_LIMIT gives the sampled policy room
// to actually recover before the clock runs out.

#define LOOP_STALL_STEPS (STARVE_LIMIT / 4)

ACTION test_time_action(SnakeENV* env, matrix* probs) {
    if (env->steps_since_food >= LOOP_STALL_STEPS) {
        return sample_action(probs); // likely stuck in a deterministic loop -- shake it loose
    }
    return greedy_action(probs);
}

// One-hot encode snake, food, and orientation into input tensor
//
// BUG FIXED HERE: the y-coordinate bounds checks below used to compare
// against `cols` instead of `rows`. It never bit anyone because
// create_env() only ever builds square grids (rows == cols), but it was a
// landmine for the day someone passes a rectangular grid_size -- the y
// check would silently pass or fail using the wrong axis's size, letting
// an out-of-bounds y slip through (or a valid one get rejected), corrupting
// the one-hot input the network sees. Taking `rows` as its own parameter
// makes the function correct regardless of grid shape.
void build_state_vector(
    matrix* in, State state, State food_state,
    ACTION pov, u32 rows, u32 cols, u32 grid_size
) {
    clear(in);

    if (state.x >= 0 && state.x < (i32)cols && state.y >= 0 && state.y < (i32)rows) {
        u32 snake_i = (u32)state.y * cols + (u32)state.x;
        in->data[snake_i] = 1.0f;
    }
    if (food_state.x >= 0 && food_state.x < (i32)cols && food_state.y >= 0 && food_state.y < (i32)rows) {
        u32 food_i = (u32)food_state.y * cols + (u32)food_state.x;
        in->data[grid_size + food_i] = 1.0f;
    }
    in->data[2 * grid_size + (u32)pov] = 1.0f;
}

// Render the snake grid to terminal with colors
void render_env(SnakeENV* env) {
    printf("\033[H"); // Move cursor to top-left
    printf("Score: %5.1f | Food Eaten: %3u\n+", env->score, env->foods_eaten);
    for (u32 c = 0; c < env->cols; c++) printf("--");
    printf("-+\n");

    for (i32 r = (i32)env->rows - 1; r >= 0; r--) {
        printf("| ");
        for (i32 c = 0; c < (i32)env->cols; c++) {
            b32 drawn = false;
            if (env->snake.x == c && env->snake.y == r) {
                printf("\033[1;32m@ \033[0m"); // Bright green head
                drawn = true;
            } else {
                for (u32 i = 1; i < env->length && !drawn; i++) {
                    if (env->body[i].x == c && env->body[i].y == r) {
                        printf("\033[0;32mo \033[0m"); // Dim green body
                        drawn = true;
                    }
                }
            }
            if (!drawn && env->food.x == c && env->food.y == r) {
                printf("\033[1;31m* \033[0m"); // Red food
                drawn = true;
            }
            if (!drawn) {
                printf(". ");
            }
        }
        printf("|\n");
    }
    printf("+");
    for (u32 c = 0; c < env->cols; c++) printf("--");
    printf("-+\n");
    
    fflush(stdout); // Force drawing frame
}

// Visual test / inference loop
void run_test_agent(model_state* model, SnakeENV* env, u32 episodes) {
    printf("\033[2J"); // Clear screen
    
    for (u32 ep = 0; ep < episodes; ep++) {
        reset_state(env);

        for (u32 step = 0; step < 100; step++) {
            render_env(env);
            sleep_ms(300);

            build_state_vector(
                model->input->val, env->snake, env->food,
                env->pov, env->rows, env->cols, env->grid_size
            );

            forward_pass(&model->forward_graph);
            ACTION action = test_time_action(env, model->output->val);
            
            State old_snake = env->snake; // Capture before moving
            take_action(env, action);

            f32 r = get_reward(env, old_snake);
            env->score += r;

            if (game_over(env)) {
                printf("\033[2J\033[H"); 
                printf("Game Over! Crashed.\nSteps Survived: %u | Foods Eaten: %u\n", step + 1, env->foods_eaten);
                sleep_ms(2000);
                break;
            }
        }
    }
}

// Save learned weights to a binary file
void save_weights(model_state* model, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    for (u32 i = 0; i < model->cost_graph.size; i++) {
        Var* cur = model->cost_graph.vars[i];
        if (cur->flags & VAR_FLAG_PARAMETER) {
            u64 size = (u64)cur->val->rows * cur->val->cols;
            fwrite(cur->val->data, sizeof(f32), size, f);
        }
    }
    fclose(f);
}

// Load weights from a binary file
b32 load_weights(model_state* model, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    
    for (u32 i = 0; i < model->cost_graph.size; i++) {
        Var* cur = model->cost_graph.vars[i];
        if (cur->flags & VAR_FLAG_PARAMETER) {
            u64 size = (u64)cur->val->rows * cur->val->cols;
            
            // Check if fread successfully read 'size' elements
            if (fread(cur->val->data, sizeof(f32), size, f) != size) {
                printf("Error: Corrupted or incomplete weights file.\n");
                fclose(f);
                return false;
            }
        }
    }
    fclose(f);
    return true;
}

// Main training loop using REINFORCE policy gradient
void train(model_state* model, SnakeENV* env) {
    u32 EPOCHS = 3500;
    u32 rollout_size = 64;
    u32 episode_len = 100;
    f32 gamma = 0.99f;
    f32 learning_rate = 0.05f;
    ReplayBuffer buffer = {0};

    for (u32 epoch = 0; epoch < EPOCHS; epoch++) {
        u32 total_foods = 0;

        for (u32 i = 0; i < rollout_size; i++) {
            reset_state(env);
            Trajectory* traj = &buffer.trajectories[i];
            traj->len = 0;

            for (u32 t = 0; t < episode_len; t++) {
                State state = env->snake;
                State food_state = env->food;
                ACTION pov = env->pov;

                build_state_vector(
                    model->input->val, state, food_state,
                    pov, env->rows, env->cols, env->grid_size
                );

                forward_pass(&model->forward_graph);
                ACTION action = sample_action(model->output->val);
                
                State old_snake = env->snake; // Capture before moving
                take_action(env, action);

                f32 reward = get_reward(env, old_snake);
                env->score += reward;
                b32 done = game_over(env);

                traj->states[t] = state;
                traj->food_states[t] = food_state;
                traj->povs[t] = pov;
                traj->actions[t] = action;
                traj->rewards[t] = reward;
                traj->len = t + 1;

                if (done) break;
            }
            total_foods += env->foods_eaten;
        }

        buffer.count = rollout_size;
        u32 sample_count = 0;
        f32 return_sum = 0.0f;
        f32 return_sq_sum = 0.0f;
        f32 episode_return_sum = 0.0f;

        for (u32 b = 0; b < buffer.count; b++) {
            Trajectory* traj = &buffer.trajectories[b];
            f32 G = 0.0f;

            for (i32 t = (i32)traj->len - 1; t >= 0; t--) {
                G = traj->rewards[t] + gamma * G;
                traj->returns[t] = G;
                return_sum += G;
                return_sq_sum += G * G;
                sample_count++;
            }
            episode_return_sum += traj->returns[0];
        }

        f32 return_mean = return_sum / (f32)sample_count;
        f32 return_var = (return_sq_sum / (f32)sample_count) - (return_mean * return_mean);
        f32 return_std = sqrtf(MAX(return_var, 1e-6f));

        for (u32 i = 0; i < model->cost_graph.size; i++) {
            Var* cur = model->cost_graph.vars[i];
            if (cur->flags & VAR_FLAG_PARAMETER) {
                clear(cur->grad);
            }
        }

        for (u32 b = 0; b < buffer.count; b++) {
            Trajectory* traj = &buffer.trajectories[b];

            for (u32 t = 0; t < traj->len; t++) {
                build_state_vector(
                    model->input->val, traj->states[t], traj->food_states[t],
                    traj->povs[t], env->rows, env->cols, env->grid_size
                );

                clear(model->advantage->val);
                f32 advantage = (traj->returns[t] - return_mean) / (return_std + 1e-8f);
                model->advantage->val->data[traj->actions[t]] = advantage;

                forward_pass(&model->cost_graph);
                backward_pass(&model->cost_graph);
            }
        }

        if (sample_count > 0) {
            f32 gradient_scale = learning_rate / (f32)sample_count;

            for (u32 i = 0; i < model->cost_graph.size; i++) {
                Var* cur = model->cost_graph.vars[i];
                if ((cur->flags & VAR_FLAG_PARAMETER) == 0) continue;

                scale(cur->grad, gradient_scale);
                sub(cur->val, cur->val, cur->grad);
            }
        }

        if (epoch % 10 == 0 || epoch == EPOCHS - 1) {
            printf(
                "Epoch %4u | Avg Return: %7.2f | Foods: %3u | Samples: %5u\n",
                epoch, episode_return_sum / (f32)buffer.count, total_foods, sample_count
            );
        }
    }
}

int main(int argc, char** argv) {
    prng_seed(1337, 42); // Seed PRNG for reproducibility

    mem_arena* arena = arena_create(GiB(1));
    model_state* model = PUSH_STRUCT(arena, model_state);

    create_actor_model(arena, model);
    model->forward_graph = build_graph(arena, model, model->output);
    model->cost_graph = build_graph(arena, model, model->cost);

    SnakeENV* env = create_env(36); // 6x6 Grid

    // Check if the user ran `./env play`
    b32 play_mode = (argc > 1 && strcmp(argv[1], "play") == 0);

    if (play_mode) {
        if (load_weights(model, "snake_weights.bin")) {
            printf("Loaded saved weights! Running Live Test Mode...\n");
            sleep_ms(1000);
            run_test_agent(model, env, 10);
        } else {
            printf("Error: No saved weights found. Please run normally first to train.\n");
        }
    } else {
        printf("=== Training Snake REINFORCE Agent ===\n");
        train(model, env);
        
        save_weights(model, "snake_weights.bin");

        printf("\n=== Training Complete. Running Live Test Mode ===\n");
        run_test_agent(model, env, 5);
    }

    arena_destroy(arena);
    destroy_env(env);
    return 0;
}