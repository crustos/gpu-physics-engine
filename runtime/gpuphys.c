/* gpuphys.c -- see gpuphys.h. The kernels (gpk_*.c) are Crust's C translation of kernels/NAME.rs. */
#include "gpuphys.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- the kernels (translated from Rust) -------------------------------------------------- */
void build_cell_ids_array( float *positions, unsigned int *cell_ids, unsigned int *object_ids, float *radius,
						   float cell_size, unsigned int num_particles );
void count_objects_for_each_chunk( unsigned int *chunk_obj_count, unsigned int *uniform_data, unsigned int *cell_ids );
void build_collision_cells_array( unsigned int *chunk_obj_count, unsigned int *collision_cells,
								  unsigned int *indirect_args, unsigned int *uniform_data, unsigned int *cell_ids );
void solve_collisions_grouped( unsigned int *chunk_obj_count, unsigned int *collision_cells, unsigned int *cell_ids,
							   unsigned int *object_ids, float *positions, float *radius, unsigned int *uniform_data,
							   unsigned int *groups, unsigned int color );
void integrate_tagged( float *positions, float *previous, float *radius, unsigned int *tags, float dt, float gx,
					   float gy, float damping, float sand_damping, float ww, float wh, unsigned int n );
void tile_collide( float *positions, float *radius, unsigned int *tiles, unsigned int tw, unsigned int th,
				   unsigned int n );
void tile_friction( float *positions, float *previous, float *radius, unsigned int *tiles, unsigned int tw,
					unsigned int th, unsigned int n, float mu, float absorb );
void solve_joints( float *positions, float *radius, unsigned int *joint_i, float *joint_f, unsigned int first,
				   unsigned int count );
void raycast( float *positions, float *radius, unsigned int *groups, unsigned int *tiles, float *rays, float *hits,
			  unsigned int *hit_ids, unsigned int tw, unsigned int th, unsigned int ncircles, unsigned int nrays,
			  unsigned int mask, unsigned int ignore_group );

/* The kernels' intrinsics, run sequentially: GP_LAUNCH sets the invocation id and calls. */
static unsigned gp__gid;
unsigned gpu_global_id( int d )
{
	return d == 0 ? gp__gid : 0u;
}
unsigned gpu_atomic_add( unsigned *p, unsigned v )
{
	unsigned old = *p;
	*p = old + v;
	return old;
}
#define GP_LAUNCH( count, CALL )                                                                                       \
	do                                                                                                                 \
	{                                                                                                                  \
		unsigned n_ = (unsigned)( count );                                                                             \
		for ( gp__gid = 0; gp__gid < n_; ++gp__gid )                                                                   \
		{                                                                                                              \
			CALL;                                                                                                      \
		}                                                                                                              \
	}                                                                                                                  \
	while ( 0 )

#define MAX_CELLS 4
#define CELL_SIZE_MULTIPLIER 2.2f
#define FIXED 0xFFFFFFFFu
#define UNUSED 0xFFFFFFFFu

typedef struct Joint
{
	int a, b, flags;
	float rest, k, fx, fy;
} Joint;

typedef struct Body
{
	int *c;
	int nc, cols, rows, active, is_box, locked;
	float w, h; /* world units */
	float *lx, *ly; /* each circle's offset from the centre when unturned, in tile units */
} Body;

typedef struct Pair
{
	unsigned key, obj, seq;
} Pair;

struct GpWorld
{
	int cap, n, max_joints, nj, nbodies, body_cap;
	float ox, oy, cell, ww, wh, gx, gy, max_radius;
	float damping, sand_damping, friction, bounce_loss;
	int iterations;
	float *pos, *prev, *rad;
	unsigned *tags, *groups;
	int *owner;
	unsigned char *frozen, *pinned;
	float *frozen_pos, *saved_rad, *pin_pos;
	unsigned *cell_ids, *object_ids, *chunks, *cells, uniform[4];
	Pair *tmp;
	unsigned *tiles;
	int tw, th;
	Joint *joints;
	unsigned *ji;
	float *jf;
	int *batch_first, *batch_count, nbatches;
	int joints_dirty;
	Body *bodies;
	int pairs, last_total, last_ncells;
};

static int cmp_pair( const void *pa, const void *pb )
{
	const Pair *a = pa, *b = pb;
	if ( a->key != b->key )
		return a->key < b->key ? -1 : 1;
	return a->seq < b->seq ? -1 : ( a->seq > b->seq );
}

GpWorld *gp_create( int capacity, int max_joints, float ox, float oy, float width, float height, float cell,
					float gravity_x, float gravity_y )
{
	GpWorld *w = calloc( 1, sizeof *w );
	if ( !w )
		return 0;
	w->cap = capacity;
	w->max_joints = max_joints;
	w->ox = ox;
	w->oy = oy;
	w->cell = cell;
	w->ww = width / cell;
	w->wh = height / cell;
	w->gx = gravity_x / cell;
	w->gy = gravity_y / cell;
	w->damping = 0.999f;
	w->sand_damping = 0.985f;
	w->friction = 0.15f;
	w->bounce_loss = 0.6f;
	w->iterations = 2;
	w->pairs = MAX_CELLS * capacity;
	int chunks = ( w->pairs + 3 ) / 4;
	w->pos = calloc( 2 * capacity, sizeof( float ) );
	w->prev = calloc( 2 * capacity, sizeof( float ) );
	w->rad = calloc( capacity, sizeof( float ) );
	w->tags = calloc( capacity, sizeof( unsigned ) );
	w->groups = calloc( capacity, sizeof( unsigned ) );
	w->owner = calloc( capacity, sizeof( int ) );
	w->frozen = calloc( capacity, 1 );
	w->pinned = calloc( capacity, 1 );
	w->pin_pos = calloc( 2 * capacity, sizeof( float ) );
	w->frozen_pos = calloc( 2 * capacity, sizeof( float ) );
	w->saved_rad = calloc( capacity, sizeof( float ) );
	w->cell_ids = calloc( w->pairs, sizeof( unsigned ) );
	w->object_ids = calloc( w->pairs, sizeof( unsigned ) );
	w->chunks = calloc( chunks + 1, sizeof( unsigned ) );
	w->cells = calloc( w->pairs + 1, sizeof( unsigned ) );
	w->tmp = calloc( w->pairs, sizeof( Pair ) );
	w->joints = calloc( max_joints > 0 ? max_joints : 1, sizeof( Joint ) );
	w->ji = calloc( 4 * ( max_joints > 0 ? max_joints : 1 ), sizeof( unsigned ) );
	w->jf = calloc( 4 * ( max_joints > 0 ? max_joints : 1 ), sizeof( float ) );
	w->batch_first = calloc( max_joints > 0 ? max_joints : 1, sizeof( int ) );
	w->batch_count = calloc( max_joints > 0 ? max_joints : 1, sizeof( int ) );
	w->body_cap = capacity;
	w->bodies = calloc( capacity, sizeof( Body ) );
	w->max_radius = 0.0f;
	return w;
}

void gp_destroy( GpWorld *w )
{
	if ( !w )
		return;
	for ( int i = 0; i < w->nbodies; ++i )
		free( w->bodies[i].c ), free( w->bodies[i].lx ), free( w->bodies[i].ly );
	free( w->pos ), free( w->prev ), free( w->rad ), free( w->tags ), free( w->groups ), free( w->owner );
	free( w->frozen ), free( w->pinned ), free( w->pin_pos ), free( w->frozen_pos ), free( w->saved_rad ), free( w->cell_ids ), free( w->object_ids ), free( w->chunks );
	free( w->cells ), free( w->tmp ), free( w->tiles ), free( w->joints ), free( w->ji ), free( w->jf );
	free( w->batch_first ), free( w->batch_count ), free( w->bodies );
	free( w );
}

void gp_set_tiles( GpWorld *w, int width, int height, const unsigned *cells )
{
	free( w->tiles );
	w->tiles = malloc( sizeof( unsigned ) * (size_t)width * height + 4 );
	memcpy( w->tiles, cells, sizeof( unsigned ) * (size_t)width * height );
	w->tw = width;
	w->th = height;
}

/* ---- circles and bodies -------------------------------------------------------------------- */
static int add_circle( GpWorld *w, float x, float y, float r, float vx, float vy, float dt, unsigned tag,
					   unsigned group, int body )
{
	if ( w->n >= w->cap )
		return -1;
	int i = w->n++;
	float c = w->cell;
	float px = ( x - w->ox ) / c, py = ( y - w->oy ) / c, ri = r / c;
	w->pos[2 * i] = px;
	w->pos[2 * i + 1] = py;
	w->prev[2 * i] = px - vx * dt / c;
	w->prev[2 * i + 1] = py - vy * dt / c;
	w->rad[i] = ri;
	w->tags[i] = tag;
	w->groups[i] = group;
	w->owner[i] = body;
	w->frozen[i] = 0;
	w->pinned[i] = 0;
	if ( ri > w->max_radius )
		w->max_radius = ri;
	return i;
}

static int new_body( GpWorld *w, int ncircles )
{
	if ( w->nbodies >= w->body_cap )
		return -1;
	Body *b = &w->bodies[w->nbodies];
	memset( b, 0, sizeof *b );
	b->c = calloc( ncircles, sizeof( int ) );
	b->active = 1;
	return w->nbodies++;
}

int gp_add_joint( GpWorld *w, int a, int b, int kind, float rest, float stiffness, float fx, float fy )
{
	if ( w->nj >= w->max_joints )
		return -1;
	Joint *j = &w->joints[w->nj];
	j->a = a;
	j->b = b;
	j->flags = kind == GP_ROPE ? 1 : kind == GP_STRUT ? 2 : 0;
	j->rest = rest / w->cell;
	j->k = stiffness;
	j->fx = ( fx - w->ox ) / w->cell;
	j->fy = ( fy - w->oy ) / w->cell;
	w->joints_dirty = 1;
	return w->nj++;
}

int gp_add_ball( GpWorld *w, float x, float y, float radius, float vx, float vy, float dt )
{
	if ( w->n >= w->cap )
		return -1;
	int id = new_body( w, 1 );
	if ( id < 0 )
		return -1;
	Body *b = &w->bodies[id];
	b->nc = 1;
	b->c[0] = add_circle( w, x, y, radius, vx, vy, dt, GP_TAG_BALL, id + 1, id );
	b->w = b->h = 2 * radius;
	return id;
}

int gp_add_box( GpWorld *w, float x, float y, float bw, float bh, float angle, float vx, float vy, float dt,
				float stiffness )
{
	float r0 = fminf( bw, bh ) / 4.0f;
	int rows = (int)floorf( bh / ( 2 * r0 ) + 0.5f ), cols = (int)floorf( bw / ( 2 * r0 ) + 0.5f );
	if ( rows < 2 )
		rows = 2;
	if ( cols < 2 )
		cols = 2;
	float r = fminf( bw / ( 2.0f * cols ), bh / ( 2.0f * rows ) );
	if ( w->n + rows * cols > w->cap || w->nj + 6 * rows * cols > w->max_joints )
		return -1;
	int id = new_body( w, rows * cols );
	if ( id < 0 )
		return -1;
	Body *b = &w->bodies[id];
	b->nc = rows * cols;
	b->cols = cols;
	b->rows = rows;
	b->is_box = 1;
	b->w = bw;
	b->h = bh;
	b->lx = calloc( rows * cols, sizeof( float ) );
	b->ly = calloc( rows * cols, sizeof( float ) );
	float ca = cosf( angle ), sa = sinf( angle );
	for ( int j = 0; j < rows; ++j )
		for ( int i = 0; i < cols; ++i )
		{
			float lx = -bw / 2 + r + i * ( bw - 2 * r ) / ( cols - 1 );
			float ly = -bh / 2 + r + j * ( bh - 2 * r ) / ( rows - 1 );
			b->lx[j * cols + i] = lx / w->cell;
			b->ly[j * cols + i] = ly / w->cell;
			b->c[j * cols + i] =
				add_circle( w, x + lx * ca - ly * sa, y + lx * sa + ly * ca, r, vx, vy, dt, GP_TAG_CRATE, id + 1, id );
		}
	for ( int a = 0; a < b->nc; ++a )
		for ( int q = a + 1; q < b->nc; ++q )
		{
			int di = abs( a % cols - q % cols ), dj = abs( a / cols - q / cols );
			if ( ( di > dj ? di : dj ) <= 2 )
			{
				float dx = w->pos[2 * b->c[a]] - w->pos[2 * b->c[q]], dy = w->pos[2 * b->c[a] + 1] - w->pos[2 * b->c[q] + 1];
				gp_add_joint( w, b->c[a], b->c[q], GP_RIGID, sqrtf( dx * dx + dy * dy ) * w->cell, stiffness, 0, 0 );
			}
		}
	return id;
}

int gp_add_rope( GpWorld *w, float ax, float ay, float bx, float by, int segments, float radius, int pin_a, int pin_b )
{
	int n = segments + 1;
	if ( w->n + n > w->cap || w->nj + n + 2 > w->max_joints )
		return -1;
	int id = new_body( w, n );
	if ( id < 0 )
		return -1;
	Body *b = &w->bodies[id];
	b->nc = n;
	for ( int k = 0; k < n; ++k )
		b->c[k] = add_circle( w, ax + ( bx - ax ) * k / segments, ay + ( by - ay ) * k / segments, radius, 0, 0, 1,
							  GP_TAG_ROPE, id + 1, id );
	float seg = sqrtf( ( bx - ax ) * ( bx - ax ) + ( by - ay ) * ( by - ay ) ) / segments;
	for ( int k = 0; k < segments; ++k )
		gp_add_joint( w, b->c[k], b->c[k + 1], GP_RIGID, seg, 1.0f, 0, 0 );
	if ( pin_a )
		gp_add_joint( w, b->c[0], -1, GP_RIGID, 0, 1.0f, ax, ay );
	if ( pin_b )
		gp_add_joint( w, b->c[n - 1], -1, GP_RIGID, 0, 1.0f, bx, by );
	return id;
}

int gp_body_count( const GpWorld *w )
{
	return w->nbodies;
}
int gp_body_circles( const GpWorld *w, int body )
{
	return w->bodies[body].nc;
}
int gp_body_circle( const GpWorld *w, int body, int k )
{
	return w->bodies[body].c[k];
}
float gp_circle_x( const GpWorld *w, int i )
{
	return w->ox + w->pos[2 * i] * w->cell;
}
float gp_circle_y( const GpWorld *w, int i )
{
	return w->oy + w->pos[2 * i + 1] * w->cell;
}
float gp_circle_radius( const GpWorld *w, int i )
{
	return w->rad[i] * w->cell;
}
int gp_circle_count( const GpWorld *w )
{
	return w->n;
}
int gp_circle_body( const GpWorld *w, int i )
{
	return w->owner[i];
}

void gp_body_pose( const GpWorld *w, int body, float *x, float *y, float *angle )
{
	const Body *b = &w->bodies[body];
	float sx = 0, sy = 0;
	for ( int k = 0; k < b->nc; ++k )
	{
		sx += w->pos[2 * b->c[k]];
		sy += w->pos[2 * b->c[k] + 1];
	}
	*x = w->ox + sx / b->nc * w->cell;
	*y = w->oy + sy / b->nc * w->cell;
	*angle = 0.0f;
	if ( b->is_box )
		*angle = atan2f( w->pos[2 * b->c[1] + 1] - w->pos[2 * b->c[0] + 1], w->pos[2 * b->c[1]] - w->pos[2 * b->c[0]] );
}

void gp_body_velocity( const GpWorld *w, int body, float dt, float *vx, float *vy )
{
	const Body *b = &w->bodies[body];
	float sx = 0, sy = 0;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		sx += w->pos[2 * c] - w->prev[2 * c];
		sy += w->pos[2 * c + 1] - w->prev[2 * c + 1];
	}
	*vx = sx / b->nc * w->cell / dt;
	*vy = sy / b->nc * w->cell / dt;
}

void gp_body_add_velocity( GpWorld *w, int body, float dt, float dvx, float dvy )
{
	const Body *b = &w->bodies[body];
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		w->prev[2 * c] -= dvx * dt / w->cell;
		w->prev[2 * c + 1] -= dvy * dt / w->cell;
	}
}

void gp_body_set_velocity( GpWorld *w, int body, float dt, float vx, float vy )
{
	float cx, cy;
	gp_body_velocity( w, body, dt, &cx, &cy );
	gp_body_add_velocity( w, body, dt, vx - cx, vy - cy );
}

void gp_body_set_position( GpWorld *w, int body, float x, float y )
{
	float cx, cy, a;
	gp_body_pose( w, body, &cx, &cy, &a );
	Body *b = &w->bodies[body];
	float dx = ( x - cx ) / w->cell, dy = ( y - cy ) / w->cell;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		w->pos[2 * c] += dx, w->pos[2 * c + 1] += dy;
		w->prev[2 * c] += dx, w->prev[2 * c + 1] += dy;
	}
}

void gp_body_lock_rotation( GpWorld *w, int body, int lock )
{
	w->bodies[body].locked = lock != 0 && w->bodies[body].is_box;
}

void gp_body_set_active( GpWorld *w, int body, int active )
{
	Body *b = &w->bodies[body];
	if ( ( b->active != 0 ) == ( active != 0 ) )
		return;
	b->active = active != 0;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		w->frozen[c] = !active;
		if ( !active )
		{ /* out of the simulation: a speck that stays where it was */
			w->frozen_pos[2 * c] = w->pos[2 * c];
			w->frozen_pos[2 * c + 1] = w->pos[2 * c + 1];
			w->saved_rad[c] = w->rad[c];
			w->rad[c] = 1.0e-4f;
		}
		else
		{
			w->rad[c] = w->saved_rad[c];
		}
	}
}

/* ---- the step ------------------------------------------------------------------------------ */
static void upload_joints( GpWorld *w )
{
	/* greedy colouring: no circle twice in a batch */
	int *colour = malloc( sizeof( int ) * ( w->nj + 1 ) );
	unsigned long long *used = calloc( w->n + 1, sizeof( unsigned long long ) );
	int maxc = 0;
	for ( int j = 0; j < w->nj; ++j )
	{
		unsigned long long taken = used[w->joints[j].a];
		if ( w->joints[j].b >= 0 )
			taken |= used[w->joints[j].b];
		int c = 0;
		while ( c < 63 && ( taken >> c & 1ull ) )
			++c;
		colour[j] = c;
		used[w->joints[j].a] |= 1ull << c;
		if ( w->joints[j].b >= 0 )
			used[w->joints[j].b] |= 1ull << c;
		if ( c > maxc )
			maxc = c;
	}
	int k = 0;
	w->nbatches = 0;
	for ( int c = 0; c <= maxc; ++c )
	{
		int first = k;
		for ( int j = 0; j < w->nj; ++j )
			if ( colour[j] == c )
			{
				Joint *q = &w->joints[j];
				w->ji[4 * k] = q->a;
				w->ji[4 * k + 1] = q->b < 0 ? FIXED : (unsigned)q->b;
				w->ji[4 * k + 2] = q->flags;
				w->ji[4 * k + 3] = 0;
				w->jf[4 * k] = q->rest;
				w->jf[4 * k + 1] = q->k;
				w->jf[4 * k + 2] = q->fx;
				w->jf[4 * k + 3] = q->fy;
				++k;
			}
		if ( k > first )
		{
			w->batch_first[w->nbatches] = first;
			w->batch_count[w->nbatches++] = k - first;
		}
	}
	free( colour );
	free( used );
	w->joints_dirty = 0;
}

static void build_cells( GpWorld *w )
{
	int n = w->n, pairs = MAX_CELLS * n;
	float cs = w->max_radius * CELL_SIZE_MULTIPLIER;
	GP_LAUNCH( n, build_cell_ids_array( w->pos, w->cell_ids, w->object_ids, w->rad, cs, n ) );
	for ( int i = 0; i < pairs; ++i )
	{
		w->tmp[i].key = w->cell_ids[i];
		w->tmp[i].obj = w->object_ids[i];
		w->tmp[i].seq = i;
	}
	qsort( w->tmp, pairs, sizeof( Pair ), cmp_pair ); /* the GPU's radix sort, stable */
	for ( int i = 0; i < pairs; ++i )
	{
		w->cell_ids[i] = w->tmp[i].key;
		w->object_ids[i] = w->tmp[i].obj;
	}
	int chunks = ( pairs + 3 ) / 4;
	w->uniform[0] = chunks;
	w->uniform[1] = pairs;
	GP_LAUNCH( chunks, count_objects_for_each_chunk( w->chunks, w->uniform, w->cell_ids ) );
	for ( int i = 1; i < chunks; ++i ) /* the prefix sum (workgroup memory on the GPU) */
		w->chunks[i] += w->chunks[i - 1];
	unsigned indirect[3];
	GP_LAUNCH( chunks, build_collision_cells_array( w->chunks, w->cells, indirect, w->uniform, w->cell_ids ) );
	w->last_total = pairs;
	w->last_ncells = (int)w->chunks[chunks - 1];
}

/* A kinematic body's circles are held where the caller put them, after every solver pass, so
 * what touches them is pushed out instead of sharing the correction */
static void restore_pinned( GpWorld *w )
{
	for ( int i = 0; i < w->n; ++i )
		if ( w->pinned[i] )
		{
			w->pos[2 * i] = w->prev[2 * i] = w->pin_pos[2 * i];
			w->pos[2 * i + 1] = w->prev[2 * i + 1] = w->pin_pos[2 * i + 1];
		}
}

void gp_body_set_kinematic( GpWorld *w, int body, int on )
{
	Body *b = &w->bodies[body];
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		w->pinned[c] = on != 0;
		w->pin_pos[2 * c] = w->pos[2 * c], w->pin_pos[2 * c + 1] = w->pos[2 * c + 1];
	}
}

void gp_body_move( GpWorld *w, int body, float x, float y )
{
	Body *b = &w->bodies[body];
	float cx = 0, cy = 0;
	for ( int k = 0; k < b->nc; ++k )
		cx += w->pin_pos[2 * b->c[k]], cy += w->pin_pos[2 * b->c[k] + 1];
	float dx = ( x - w->ox ) / w->cell - cx / b->nc, dy = ( y - w->oy ) / w->cell - cy / b->nc;
	for ( int k = 0; k < b->nc; ++k )
		w->pin_pos[2 * b->c[k]] += dx, w->pin_pos[2 * b->c[k] + 1] += dy;
}

void gp_set_material( GpWorld *w, float friction, float bounciness )
{
	w->friction = 0.375f * friction; /* the kernel's mu: 0.15 is Unity's default friction 0.4 */
	w->bounce_loss = 1.0f - ( bounciness < 0 ? 0 : ( bounciness > 1 ? 1 : bounciness ) );
}

void gp_step( GpWorld *w, float dt )
{
	int n = w->n;
	if ( n == 0 )
		return;
	if ( w->joints_dirty )
		upload_joints( w );
	GP_LAUNCH( n, integrate_tagged( w->pos, w->prev, w->rad, w->tags, dt, w->gx, w->gy, w->damping, w->sand_damping,
									w->ww, w->wh, n ) );
	restore_pinned( w );
	build_cells( w );
	for ( int it = 0; it < ( w->iterations > 0 ? w->iterations : 1 ); ++it )
	{
		for ( unsigned color = 1; color <= 4; ++color )
			GP_LAUNCH( w->last_ncells,
					   solve_collisions_grouped( w->chunks, w->cells, w->cell_ids, w->object_ids, w->pos, w->rad,
												 w->uniform, w->groups, color ) );
		restore_pinned( w );
		for ( int b = 0; b < w->nbatches; ++b )
			GP_LAUNCH( w->batch_count[b], solve_joints( w->pos, w->rad, w->ji, w->jf, w->batch_first[b], w->batch_count[b] ) );
		if ( w->tw && w->th )
			GP_LAUNCH( n, tile_collide( w->pos, w->rad, w->tiles, w->tw, w->th, n ) );
		restore_pinned( w );
	}
	if ( w->tw && w->th && ( w->friction > 0 || w->bounce_loss > 0 ) )
		GP_LAUNCH( n, tile_friction( w->pos, w->prev, w->rad, w->tiles, w->tw, w->th, n, w->friction, w->bounce_loss ) );
	for ( int bi = 0; bi < w->nbodies; ++bi )
	{
		Body *b = &w->bodies[bi];
		if ( !b->locked || !b->active )
			continue;
		float cx = 0, cy = 0;
		for ( int k = 0; k < b->nc; ++k )
			cx += w->pos[2 * b->c[k]], cy += w->pos[2 * b->c[k] + 1];
		cx /= b->nc, cy /= b->nc;
		for ( int k = 0; k < b->nc; ++k )
		{
			int c = b->c[k];
			float nx = cx + b->lx[k], ny = cy + b->ly[k];
			w->prev[2 * c] += nx - w->pos[2 * c];
			w->prev[2 * c + 1] += ny - w->pos[2 * c + 1];
			w->pos[2 * c] = nx, w->pos[2 * c + 1] = ny;
		}
	}
	for ( int i = 0; i < n; ++i )
		if ( w->frozen[i] )
		{
			w->pos[2 * i] = w->prev[2 * i] = w->frozen_pos[2 * i];
			w->pos[2 * i + 1] = w->prev[2 * i + 1] = w->frozen_pos[2 * i + 1];
		}
}

/* ---- queries ------------------------------------------------------------------------------- */
int gp_raycast( GpWorld *w, float ox, float oy, float dx, float dy, float distance, int mask, int ignore_body,
				GpHit *hit )
{
	float len = sqrtf( dx * dx + dy * dy );
	if ( len <= 0.0f )
		return 0;
	float ray[8] = { ( ox - w->ox ) / w->cell, ( oy - w->oy ) / w->cell, dx / len, dy / len, distance / w->cell, 0, 0, 0 };
	float h[4];
	unsigned id = 0;
	/* with no tile grid the kernel must not walk one */
	GP_LAUNCH( 1, raycast( w->pos, w->rad, w->groups, w->tiles, ray, h, &id, w->tw, w->th, w->n, 1, mask,
						   ignore_body >= 0 ? (unsigned)ignore_body + 1u : 0u ) );
	if ( h[3] == 0.0f )
		return 0;
	hit->distance = h[0] * w->cell;
	hit->x = ox + dx / len * hit->distance;
	hit->y = oy + dy / len * hit->distance;
	hit->nx = h[1];
	hit->ny = h[2];
	hit->kind = (int)h[3];
	hit->body = -1;
	hit->tile = -1;
	if ( hit->kind == 2 )
		hit->body = w->owner[id];
	else
		hit->tile = (int)id;
	return 1;
}

int gp_overlap_circle( const GpWorld *w, float x, float y, float radius, int *body )
{
	for ( int i = 0; i < w->n; ++i )
	{
		float dx = gp_circle_x( w, i ) - x, dy = gp_circle_y( w, i ) - y, r = gp_circle_radius( w, i ) + radius;
		if ( dx * dx + dy * dy < r * r )
		{
			if ( body )
				*body = w->owner[i];
			return 1;
		}
	}
	return 0;
}

int gp_tile_at( const GpWorld *w, float x, float y, int *tile )
{
	int tx = (int)floorf( ( x - w->ox ) / w->cell ), ty = (int)floorf( ( y - w->oy ) / w->cell );
	if ( tx < 0 || ty < 0 || tx >= w->tw || ty >= w->th )
		return 0;
	if ( tile )
		*tile = ty * w->tw + tx;
	return w->tiles[ty * w->tw + tx] != 0;
}

int gp_body_pairs( const GpWorld *w, int *a, int *b, int max )
{
	int n = 0;
	for ( int i = 0; i < w->n; ++i )
		for ( int j = i + 1; j < w->n; ++j )
		{
			if ( w->owner[i] == w->owner[j] || w->frozen[i] || w->frozen[j] )
				continue;
			float dx = w->pos[2 * i] - w->pos[2 * j], dy = w->pos[2 * i + 1] - w->pos[2 * j + 1];
			float r = w->rad[i] + w->rad[j] + 0.03f;
			if ( dx * dx + dy * dy < r * r )
			{
				int p = w->owner[i] < w->owner[j] ? w->owner[i] : w->owner[j];
				int q = w->owner[i] < w->owner[j] ? w->owner[j] : w->owner[i];
				int dup = 0;
				for ( int k = 0; k < n && !dup; ++k )
					dup = a[k] == p && b[k] == q;
				if ( !dup && n < max )
				{
					a[n] = p;
					b[n] = q;
					++n;
				}
			}
		}
	return n;
}

int gp_tile_contacts( const GpWorld *w, int *body, int *tile, int max )
{
	int n = 0;
	for ( int i = 0; i < w->n && w->tw; ++i )
	{
		if ( w->frozen[i] )
			continue;
		float x = w->pos[2 * i], y = w->pos[2 * i + 1], r = w->rad[i] + 0.03f;
		for ( int ty = (int)floorf( y - r ); ty <= (int)floorf( y + r ); ++ty )
			for ( int tx = (int)floorf( x - r ); tx <= (int)floorf( x + r ); ++tx )
			{
				if ( tx < 0 || ty < 0 || tx >= w->tw || ty >= w->th || !w->tiles[ty * w->tw + tx] )
					continue;
				float qx = fminf( fmaxf( x, tx ), tx + 1.0f ), qy = fminf( fmaxf( y, ty ), ty + 1.0f );
				float dx = x - qx, dy = y - qy;
				if ( dx * dx + dy * dy < r * r )
				{
					int dup = 0;
					for ( int k = 0; k < n && !dup; ++k )
						dup = body[k] == w->owner[i] && tile[k] == ty * w->tw + tx;
					if ( !dup && n < max )
					{
						body[n] = w->owner[i];
						tile[n] = ty * w->tw + tx;
						++n;
					}
				}
			}
	}
	return n;
}
