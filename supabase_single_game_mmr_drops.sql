create table if not exists public.single_game_mmr_drops (
    player_name text not null,
    region text not null check (region in ('EU', 'NA', 'AP')),
    season integer not null,
    game_time timestamptz not null,
    previous_time timestamptz not null,
    mmr_before integer not null check (mmr_before >= 16000),
    mmr_after integer not null,
    mmr_drop integer not null check (mmr_drop between 1 and 185),
    opponent_mmr numeric not null check (opponent_mmr >= 6000),
    created_at timestamptz not null default now(),
    primary key (player_name, region, game_time),
    check (mmr_before - mmr_after = mmr_drop),
    check (game_time > previous_time)
);

create index if not exists single_game_mmr_drops_season_region_drop_idx
    on public.single_game_mmr_drops (season, region, mmr_drop desc);

alter table public.single_game_mmr_drops enable row level security;

grant select on public.single_game_mmr_drops to anon, authenticated;
grant insert, update on public.single_game_mmr_drops to authenticated, service_role;

drop policy if exists "Public can read single-game MMR drops"
    on public.single_game_mmr_drops;
create policy "Public can read single-game MMR drops"
    on public.single_game_mmr_drops
    for select
    to anon, authenticated
    using (true);

drop policy if exists "Authenticated app can add single-game MMR drops"
    on public.single_game_mmr_drops;
create policy "Authenticated app can add single-game MMR drops"
    on public.single_game_mmr_drops
    for insert
    to authenticated
    with check (true);

drop policy if exists "Authenticated app can update single-game MMR drops"
    on public.single_game_mmr_drops;
create policy "Authenticated app can update single-game MMR drops"
    on public.single_game_mmr_drops
    for update
    to authenticated
    using (true)
    with check (true);
