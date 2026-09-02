#include "base.h"
#include "arena.h"
#include "prng.h"
#include "mat.h"
#include "autograd.h"

// Forward source includes for unity build
#include "arena.c"
#include "prng.c"
#include "mat.c"
#include "autograd.c"

#define INPUT_DIM 77  // 36 (snake) + 36 (food) + 5 (POV actions)
#define NUM_ACTIONS 5 // LEFT, RIGHT, UP, DOWN, NONE

// Build 2-hidden-layer MLP policy network: (77 -> 128 -> 128 -> 5)
void create_actor_model(mem_arena* arena, model_state* model) {
    // Allocate network input placeholder
    Var* input = var_create(arena, model, INPUT_DIM, 1, VAR_FLAG_NONE);
    model->input = input;

    // Layer 0: Input -> Hidden 1
    Var* W0 = var_create(arena, model, 128, INPUT_DIM, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    Var* b0 = var_create(arena, model, 128, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

    // Layer 1: Hidden 1 -> Hidden 2
    Var* W1 = var_create(arena, model, 128, 128, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    Var* b1 = var_create(arena, model, 128, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

    // Layer 2: Hidden 2 -> Action Logits
    Var* W2 = var_create(arena, model, NUM_ACTIONS, 128, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    Var* b2 = var_create(arena, model, NUM_ACTIONS, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

    // Xavier/Glorot uniform initialization bounds
    f32 bound0 = sqrtf(6.0f / (INPUT_DIM + 128));
    f32 bound1 = sqrtf(6.0f / (128 + 128));
    f32 bound2 = sqrtf(6.0f / (128 + NUM_ACTIONS));
  
    fill_rand(W0->val, -bound0, bound0);
    fill_rand(W1->val, -bound1, bound1);
    fill_rand(W2->val, -bound2, bound2);

    // Forward pass definitions
    Var* z0_a = var_matmul(arena, model, W0, input);
    Var* z0_b = var_add(arena, model, z0_a, b0);
    Var* a0 = var_relu(arena, model, z0_b);

    Var* z1_a = var_matmul(arena, model, W1, a0);
    Var* z1_b = var_add(arena, model, z1_a, b1);
    Var* a1 = var_relu(arena, model, z1_b);

    Var* z2_a = var_matmul(arena, model, W2, a1);
    Var* z2_b = var_add(arena, model, z2_a, b2);

    // Softmax policy probabilities output
    Var* output = var_softmax(arena, model, z2_b);
    model->output = output;

    // Advantage placeholder for REINFORCE policy loss
    Var* advantage = var_create(arena, model, NUM_ACTIONS, 1, VAR_FLAG_NONE);
    model->advantage = advantage;

    // REINFORCE loss node: J = -log(probs) * advantage
    Var* cost = var_reinforce_loss(arena, model, output, advantage);
    model->cost = cost;
}