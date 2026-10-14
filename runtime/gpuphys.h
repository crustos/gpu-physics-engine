/* gpuphys -- the circle physics of gpu-physics-engine as a C library.
 *
 * The same kernels the GPU runs (kernels/NAME.rs, translated to C by Crust) run here once per
 * invocation on the CPU; the parts of the pipeline that need workgroup memory (the radix sort,
 * the prefix sum) are plain C. Results match the reference interpreter's.
 *
 * Space: the library works in tile units (one solid-tile cell = 1.0) so the tile collider needs
 * no scale; every call here takes and returns WORLD units (`cell` world units per tile, `ox, oy`
 * the world position of tile (0, 0)'s corner).
 *
 * Everything is circles: a ball is one; a box is a grid of at least 2 x 2 held by stiff
 * distance joints; a rope is a chain. Circles of one body never collide with each other.
 */
#ifndef GPUPHYS_H
#define GPUPHYS_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpWorld GpWorld;

enum { GP_TAG_BALL = 0, GP_TAG_SAND = 1, GP_TAG_CRATE = 2, GP_TAG_SHARD = 3, GP_TAG_ROPE = 4 };
enum { GP_RIGID = 0, GP_ROPE = 1, GP_STRUT = 2 };
enum { GP_RAY_TILES = 1, GP_RAY_CIRCLES = 2 };

typedef struct GpHit {
	float x, y, nx, ny, distance;
	int kind;  /* 1 tile, 2 circle */
	int body;  /* the body of a circle hit, -1 for a free circle */
	int tile;  /* the tile's index (ty * tiles_w + tx) */
} GpHit;

/* A world of up to `capacity` circles and `max_joints` joints. The world box is
 * [ox, ox + width] x [oy, oy + height]; gravity in world units / s^2. */
GpWorld *gp_create( int capacity, int max_joints, float ox, float oy, float width, float height, float cell,
					float gravity_x, float gravity_y );
void gp_destroy( GpWorld *w );

/* Solid tiles: width x height cells, row-major (index = ty * width + tx), non-zero is solid; tile
 * (0, 0) is the cell at (ox, oy). The cells are copied. */
void gp_set_tiles( GpWorld *w, int width, int height, const unsigned *cells );

/* Bodies. Each returns a body id (>= 0) or -1 when full. Velocities in world units / s. */
int gp_add_ball( GpWorld *w, float x, float y, float radius, float vx, float vy, float dt );
int gp_add_box( GpWorld *w, float x, float y, float width, float height, float angle, float vx, float vy,
				float dt, float stiffness );
int gp_add_rope( GpWorld *w, float ax, float ay, float bx, float by, int segments, float radius, int pin_a, int pin_b );

int gp_body_count( const GpWorld *w );
int gp_body_circles( const GpWorld *w, int body );
int gp_body_circle( const GpWorld *w, int body, int k ); /* circle index of the k-th circle */
void gp_body_pose( const GpWorld *w, int body, float *x, float *y, float *angle );
void gp_body_velocity( const GpWorld *w, int body, float dt, float *vx, float *vy );
void gp_body_set_velocity( GpWorld *w, int body, float dt, float vx, float vy );
void gp_body_add_velocity( GpWorld *w, int body, float dt, float dvx, float dvy );
void gp_body_set_position( GpWorld *w, int body, float x, float y ); /* a teleport that keeps velocity */
void gp_body_set_active( GpWorld *w, int body, int active ); /* inactive: out of the simulation */
/* A locked box keeps its orientation (shape matching after every step): Unity's freezeRotation. */
/* Kinematic: the body's circles stay where gp_body_move puts them (pushing, never pushed). */
void gp_body_set_kinematic( GpWorld *w, int body, int on );
void gp_body_move( GpWorld *w, int body, float x, float y ); /* its centre, world units */
void gp_body_lock_rotation( GpWorld *w, int body, int lock );

/* Joints between circle indices (a_circle, b_circle), or a circle and a fixed point (b < 0). */
int gp_add_joint( GpWorld *w, int a, int b, int kind, float rest, float stiffness, float fx, float fy );

/* One surface for the world: friction (Coulomb-like, 0.4 is Unity's default) and bounciness (0..1).
 * The kernels take one pair for every contact between a circle and a tile. */
void gp_set_material( GpWorld *w, float friction, float bounciness );

void gp_step( GpWorld *w, float dt );

float gp_circle_x( const GpWorld *w, int circle );
float gp_circle_y( const GpWorld *w, int circle );
float gp_circle_radius( const GpWorld *w, int circle );
int gp_circle_count( const GpWorld *w );
int gp_circle_body( const GpWorld *w, int circle );

/* Ray cast (direction need not be unit). Returns 1 and fills *hit, or 0. ignore_body < 0: none. */
int gp_raycast( GpWorld *w, float ox, float oy, float dx, float dy, float distance, int mask, int ignore_body,
				GpHit *hit );
/* A circle (a body's circle) overlapping the disc: returns its body id (or -1 / not found: 0). */
int gp_overlap_circle( const GpWorld *w, float x, float y, float radius, int *body );
/* Tile solid at the world point, 1/0; also its index in *tile when non-null. */
int gp_tile_at( const GpWorld *w, float x, float y, int *tile );

/* Pairs of bodies whose circles touch (distance < r1 + r2 + slop) after the last step, each pair
 * once with a < b; and body-tile contacts as (body, tile). Returns how many were written. */
int gp_body_pairs( const GpWorld *w, int *a, int *b, int max );
int gp_tile_contacts( const GpWorld *w, int *body, int *tile, int max );

#ifdef __cplusplus
}
#endif
#endif
