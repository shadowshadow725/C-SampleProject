#include "hash.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include "hash.h"
#include <stdio.h>
#include <string.h>


int hash(int n, int size){
    return (int)(n % size);
}

struct _node{
    double data;
    int key;
    struct _node *next;
};

struct _hash_table_t {
	int size;// should be a prime number
    struct _node **nodes;
	//TODO
};


static bool is_prime(int n)
{
	assert(n > 0);
	for (int i = 2; i <= sqrt(n); i++) {
		if (n % i == 0) return false;
	}
	return true;
}

// Get the smallest prime number that is not less than n (for hash table size computation)
int next_prime(int n)
{
	for (int i = n; ; i++) {
		if (is_prime(i)) return i;
	}
	assert(false);
	return 0;
}


// Create a hash table with 'size' buckets; the storage is allocated dynamically using malloc(); returns NULL on error
hash_table_t *hash_create(int size)
{
	assert(size > 0);
    if(!is_prime(size)){
        size = next_prime(size);
    }

	struct _hash_table_t *table = malloc(sizeof(struct _hash_table_t ));

	table->size = size;
    table->nodes = malloc(sizeof(struct _node) * size);

    for(int i = 0;i<size;i++){
        table->nodes[i] = malloc(sizeof(node));
		table->nodes[i]->data = -1;
		table->nodes[i]->key = -1;
		table->nodes[i]->next = NULL;
    }
    
	return table;

}

int hash_put(hash_table_t *table, int key, double value){
    assert(table != NULL);
    int hsh = hash(key, table->size);
    node *local = (table->nodes[hsh]);
	if(local->key == key){
        return 0;
    }
    if(local == NULL ){
        local = malloc(sizeof(node));
        local->key = key;
        local->data = value;
        return 0;
    }
    while(local->next != NULL && local->key != key){
        local = local->next;
    }
    local->next = malloc(sizeof(node));
	local = local->next;
    local->key = key;
    local->data = value;
    local->next = NULL;
    return 0;
}

double hash_get(hash_table_t *table, int key){
    assert(table != NULL);
    int hsh = hash(key, table->size);
    node *original = (table->nodes[hsh]);
    node *local = (table->nodes[hsh]);
    while(local != NULL && local->key != key){
        local = local->next;
    }
    if (local == NULL){
        table->nodes[hsh] = original;
        return -1;
    }
    table->nodes[hsh] = original;
    return local->data;
}
void node_destory(node *this){
    while(this != NULL){
        node *local = this;
        this = this->next;
        free(local);
    }
    free(this);
}

void hash_destroy(hash_table_t *table){
    for (int i = 0;i<table->size;i++){
        node_destory(table->nodes[i]);
    }
    free(table);
    
}
