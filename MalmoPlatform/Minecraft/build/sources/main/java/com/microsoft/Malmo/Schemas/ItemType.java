//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2022.05.30 at 10:01:29 AM EEST 
//


package com.microsoft.Malmo.Schemas;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ItemType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="ItemType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="iron_shovel"/>
 *     &lt;enumeration value="iron_pickaxe"/>
 *     &lt;enumeration value="iron_axe"/>
 *     &lt;enumeration value="flint_and_steel"/>
 *     &lt;enumeration value="apple"/>
 *     &lt;enumeration value="bow"/>
 *     &lt;enumeration value="arrow"/>
 *     &lt;enumeration value="coal"/>
 *     &lt;enumeration value="diamond"/>
 *     &lt;enumeration value="iron_ingot"/>
 *     &lt;enumeration value="gold_ingot"/>
 *     &lt;enumeration value="iron_sword"/>
 *     &lt;enumeration value="wooden_sword"/>
 *     &lt;enumeration value="wooden_shovel"/>
 *     &lt;enumeration value="wooden_pickaxe"/>
 *     &lt;enumeration value="wooden_axe"/>
 *     &lt;enumeration value="stone_sword"/>
 *     &lt;enumeration value="stone_shovel"/>
 *     &lt;enumeration value="stone_pickaxe"/>
 *     &lt;enumeration value="stone_axe"/>
 *     &lt;enumeration value="diamond_sword"/>
 *     &lt;enumeration value="diamond_shovel"/>
 *     &lt;enumeration value="diamond_pickaxe"/>
 *     &lt;enumeration value="diamond_axe"/>
 *     &lt;enumeration value="stick"/>
 *     &lt;enumeration value="bowl"/>
 *     &lt;enumeration value="mushroom_stew"/>
 *     &lt;enumeration value="golden_sword"/>
 *     &lt;enumeration value="golden_shovel"/>
 *     &lt;enumeration value="golden_pickaxe"/>
 *     &lt;enumeration value="golden_axe"/>
 *     &lt;enumeration value="string"/>
 *     &lt;enumeration value="feather"/>
 *     &lt;enumeration value="gunpowder"/>
 *     &lt;enumeration value="wooden_hoe"/>
 *     &lt;enumeration value="stone_hoe"/>
 *     &lt;enumeration value="iron_hoe"/>
 *     &lt;enumeration value="diamond_hoe"/>
 *     &lt;enumeration value="golden_hoe"/>
 *     &lt;enumeration value="wheat_seeds"/>
 *     &lt;enumeration value="wheat"/>
 *     &lt;enumeration value="bread"/>
 *     &lt;enumeration value="leather_helmet"/>
 *     &lt;enumeration value="leather_chestplate"/>
 *     &lt;enumeration value="leather_leggings"/>
 *     &lt;enumeration value="leather_boots"/>
 *     &lt;enumeration value="chainmail_helmet"/>
 *     &lt;enumeration value="chainmail_chestplate"/>
 *     &lt;enumeration value="chainmail_leggings"/>
 *     &lt;enumeration value="chainmail_boots"/>
 *     &lt;enumeration value="iron_helmet"/>
 *     &lt;enumeration value="iron_chestplate"/>
 *     &lt;enumeration value="iron_leggings"/>
 *     &lt;enumeration value="iron_boots"/>
 *     &lt;enumeration value="diamond_helmet"/>
 *     &lt;enumeration value="diamond_chestplate"/>
 *     &lt;enumeration value="diamond_leggings"/>
 *     &lt;enumeration value="diamond_boots"/>
 *     &lt;enumeration value="golden_helmet"/>
 *     &lt;enumeration value="golden_chestplate"/>
 *     &lt;enumeration value="golden_leggings"/>
 *     &lt;enumeration value="golden_boots"/>
 *     &lt;enumeration value="flint"/>
 *     &lt;enumeration value="porkchop"/>
 *     &lt;enumeration value="cooked_porkchop"/>
 *     &lt;enumeration value="painting"/>
 *     &lt;enumeration value="golden_apple"/>
 *     &lt;enumeration value="sign"/>
 *     &lt;enumeration value="wooden_door"/>
 *     &lt;enumeration value="bucket"/>
 *     &lt;enumeration value="bucket"/>
 *     &lt;enumeration value="water_bucket"/>
 *     &lt;enumeration value="lava_bucket"/>
 *     &lt;enumeration value="minecart"/>
 *     &lt;enumeration value="saddle"/>
 *     &lt;enumeration value="iron_door"/>
 *     &lt;enumeration value="redstone"/>
 *     &lt;enumeration value="snowball"/>
 *     &lt;enumeration value="boat"/>
 *     &lt;enumeration value="leather"/>
 *     &lt;enumeration value="milk_bucket"/>
 *     &lt;enumeration value="brick"/>
 *     &lt;enumeration value="clay_ball"/>
 *     &lt;enumeration value="reeds"/>
 *     &lt;enumeration value="paper"/>
 *     &lt;enumeration value="book"/>
 *     &lt;enumeration value="slime_ball"/>
 *     &lt;enumeration value="chest_minecart"/>
 *     &lt;enumeration value="furnace_minecart"/>
 *     &lt;enumeration value="egg"/>
 *     &lt;enumeration value="compass"/>
 *     &lt;enumeration value="fishing_rod"/>
 *     &lt;enumeration value="clock"/>
 *     &lt;enumeration value="glowstone_dust"/>
 *     &lt;enumeration value="fish"/>
 *     &lt;enumeration value="cooked_fish"/>
 *     &lt;enumeration value="dye"/>
 *     &lt;enumeration value="bone"/>
 *     &lt;enumeration value="sugar"/>
 *     &lt;enumeration value="cake"/>
 *     &lt;enumeration value="bed"/>
 *     &lt;enumeration value="repeater"/>
 *     &lt;enumeration value="cookie"/>
 *     &lt;enumeration value="filled_map"/>
 *     &lt;enumeration value="shears"/>
 *     &lt;enumeration value="melon"/>
 *     &lt;enumeration value="pumpkin_seeds"/>
 *     &lt;enumeration value="melon_seeds"/>
 *     &lt;enumeration value="beef"/>
 *     &lt;enumeration value="cooked_beef"/>
 *     &lt;enumeration value="chicken"/>
 *     &lt;enumeration value="cooked_chicken"/>
 *     &lt;enumeration value="rotten_flesh"/>
 *     &lt;enumeration value="ender_pearl"/>
 *     &lt;enumeration value="blaze_rod"/>
 *     &lt;enumeration value="ghast_tear"/>
 *     &lt;enumeration value="gold_nugget"/>
 *     &lt;enumeration value="nether_wart"/>
 *     &lt;enumeration value="potion"/>
 *     &lt;enumeration value="glass_bottle"/>
 *     &lt;enumeration value="spider_eye"/>
 *     &lt;enumeration value="fermented_spider_eye"/>
 *     &lt;enumeration value="blaze_powder"/>
 *     &lt;enumeration value="magma_cream"/>
 *     &lt;enumeration value="brewing_stand"/>
 *     &lt;enumeration value="cauldron"/>
 *     &lt;enumeration value="ender_eye"/>
 *     &lt;enumeration value="speckled_melon"/>
 *     &lt;enumeration value="spawn_egg"/>
 *     &lt;enumeration value="experience_bottle"/>
 *     &lt;enumeration value="fire_charge"/>
 *     &lt;enumeration value="writable_book"/>
 *     &lt;enumeration value="written_book"/>
 *     &lt;enumeration value="emerald"/>
 *     &lt;enumeration value="item_frame"/>
 *     &lt;enumeration value="flower_pot"/>
 *     &lt;enumeration value="carrot"/>
 *     &lt;enumeration value="potato"/>
 *     &lt;enumeration value="baked_potato"/>
 *     &lt;enumeration value="poisonous_potato"/>
 *     &lt;enumeration value="map"/>
 *     &lt;enumeration value="golden_carrot"/>
 *     &lt;enumeration value="skull"/>
 *     &lt;enumeration value="carrot_on_a_stick"/>
 *     &lt;enumeration value="nether_star"/>
 *     &lt;enumeration value="pumpkin_pie"/>
 *     &lt;enumeration value="fireworks"/>
 *     &lt;enumeration value="firework_charge"/>
 *     &lt;enumeration value="enchanted_book"/>
 *     &lt;enumeration value="comparator"/>
 *     &lt;enumeration value="netherbrick"/>
 *     &lt;enumeration value="quartz"/>
 *     &lt;enumeration value="tnt_minecart"/>
 *     &lt;enumeration value="hopper_minecart"/>
 *     &lt;enumeration value="prismarine_shard"/>
 *     &lt;enumeration value="prismarine_crystals"/>
 *     &lt;enumeration value="rabbit"/>
 *     &lt;enumeration value="cooked_rabbit"/>
 *     &lt;enumeration value="rabbit_stew"/>
 *     &lt;enumeration value="rabbit_foot"/>
 *     &lt;enumeration value="rabbit_hide"/>
 *     &lt;enumeration value="armor_stand"/>
 *     &lt;enumeration value="iron_horse_armor"/>
 *     &lt;enumeration value="golden_horse_armor"/>
 *     &lt;enumeration value="diamond_horse_armor"/>
 *     &lt;enumeration value="lead"/>
 *     &lt;enumeration value="name_tag"/>
 *     &lt;enumeration value="command_block_minecart"/>
 *     &lt;enumeration value="mutton"/>
 *     &lt;enumeration value="cooked_mutton"/>
 *     &lt;enumeration value="banner"/>
 *     &lt;enumeration value="spruce_door"/>
 *     &lt;enumeration value="birch_door"/>
 *     &lt;enumeration value="jungle_door"/>
 *     &lt;enumeration value="acacia_door"/>
 *     &lt;enumeration value="dark_oak_door"/>
 *     &lt;enumeration value="chorus_fruit"/>
 *     &lt;enumeration value="chorus_fruit_popped"/>
 *     &lt;enumeration value="beetroot"/>
 *     &lt;enumeration value="beetroot_seeds"/>
 *     &lt;enumeration value="beetroot_soup"/>
 *     &lt;enumeration value="dragon_breath"/>
 *     &lt;enumeration value="splash_potion"/>
 *     &lt;enumeration value="spectral_arrow"/>
 *     &lt;enumeration value="tipped_arrow"/>
 *     &lt;enumeration value="lingering_potion"/>
 *     &lt;enumeration value="shield"/>
 *     &lt;enumeration value="elytra"/>
 *     &lt;enumeration value="spruce_boat"/>
 *     &lt;enumeration value="birch_boat"/>
 *     &lt;enumeration value="jungle_boat"/>
 *     &lt;enumeration value="acacia_boat"/>
 *     &lt;enumeration value="dark_oak_boat"/>
 *     &lt;enumeration value="totem_of_undying"/>
 *     &lt;enumeration value="shulker_shell"/>
 *     &lt;enumeration value="iron_nugget"/>
 *     &lt;enumeration value="record_13"/>
 *     &lt;enumeration value="record_cat"/>
 *     &lt;enumeration value="record_blocks"/>
 *     &lt;enumeration value="record_chirp"/>
 *     &lt;enumeration value="record_far"/>
 *     &lt;enumeration value="record_mall"/>
 *     &lt;enumeration value="record_mellohi"/>
 *     &lt;enumeration value="record_stal"/>
 *     &lt;enumeration value="record_strad"/>
 *     &lt;enumeration value="record_ward"/>
 *     &lt;enumeration value="record_11"/>
 *     &lt;enumeration value="record_wait"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlType(name = "ItemType")
@XmlEnum
public enum ItemType {

    @XmlEnumValue("iron_shovel")
    IRON_SHOVEL("iron_shovel"),
    @XmlEnumValue("iron_pickaxe")
    IRON_PICKAXE("iron_pickaxe"),
    @XmlEnumValue("iron_axe")
    IRON_AXE("iron_axe"),
    @XmlEnumValue("flint_and_steel")
    FLINT_AND_STEEL("flint_and_steel"),
    @XmlEnumValue("apple")
    APPLE("apple"),
    @XmlEnumValue("bow")
    BOW("bow"),
    @XmlEnumValue("arrow")
    ARROW("arrow"),
    @XmlEnumValue("coal")
    COAL("coal"),
    @XmlEnumValue("diamond")
    DIAMOND("diamond"),
    @XmlEnumValue("iron_ingot")
    IRON_INGOT("iron_ingot"),
    @XmlEnumValue("gold_ingot")
    GOLD_INGOT("gold_ingot"),
    @XmlEnumValue("iron_sword")
    IRON_SWORD("iron_sword"),
    @XmlEnumValue("wooden_sword")
    WOODEN_SWORD("wooden_sword"),
    @XmlEnumValue("wooden_shovel")
    WOODEN_SHOVEL("wooden_shovel"),
    @XmlEnumValue("wooden_pickaxe")
    WOODEN_PICKAXE("wooden_pickaxe"),
    @XmlEnumValue("wooden_axe")
    WOODEN_AXE("wooden_axe"),
    @XmlEnumValue("stone_sword")
    STONE_SWORD("stone_sword"),
    @XmlEnumValue("stone_shovel")
    STONE_SHOVEL("stone_shovel"),
    @XmlEnumValue("stone_pickaxe")
    STONE_PICKAXE("stone_pickaxe"),
    @XmlEnumValue("stone_axe")
    STONE_AXE("stone_axe"),
    @XmlEnumValue("diamond_sword")
    DIAMOND_SWORD("diamond_sword"),
    @XmlEnumValue("diamond_shovel")
    DIAMOND_SHOVEL("diamond_shovel"),
    @XmlEnumValue("diamond_pickaxe")
    DIAMOND_PICKAXE("diamond_pickaxe"),
    @XmlEnumValue("diamond_axe")
    DIAMOND_AXE("diamond_axe"),
    @XmlEnumValue("stick")
    STICK("stick"),
    @XmlEnumValue("bowl")
    BOWL("bowl"),
    @XmlEnumValue("mushroom_stew")
    MUSHROOM_STEW("mushroom_stew"),
    @XmlEnumValue("golden_sword")
    GOLDEN_SWORD("golden_sword"),
    @XmlEnumValue("golden_shovel")
    GOLDEN_SHOVEL("golden_shovel"),
    @XmlEnumValue("golden_pickaxe")
    GOLDEN_PICKAXE("golden_pickaxe"),
    @XmlEnumValue("golden_axe")
    GOLDEN_AXE("golden_axe"),
    @XmlEnumValue("string")
    STRING("string"),
    @XmlEnumValue("feather")
    FEATHER("feather"),
    @XmlEnumValue("gunpowder")
    GUNPOWDER("gunpowder"),
    @XmlEnumValue("wooden_hoe")
    WOODEN_HOE("wooden_hoe"),
    @XmlEnumValue("stone_hoe")
    STONE_HOE("stone_hoe"),
    @XmlEnumValue("iron_hoe")
    IRON_HOE("iron_hoe"),
    @XmlEnumValue("diamond_hoe")
    DIAMOND_HOE("diamond_hoe"),
    @XmlEnumValue("golden_hoe")
    GOLDEN_HOE("golden_hoe"),
    @XmlEnumValue("wheat_seeds")
    WHEAT_SEEDS("wheat_seeds"),
    @XmlEnumValue("wheat")
    WHEAT("wheat"),
    @XmlEnumValue("bread")
    BREAD("bread"),
    @XmlEnumValue("leather_helmet")
    LEATHER_HELMET("leather_helmet"),
    @XmlEnumValue("leather_chestplate")
    LEATHER_CHESTPLATE("leather_chestplate"),
    @XmlEnumValue("leather_leggings")
    LEATHER_LEGGINGS("leather_leggings"),
    @XmlEnumValue("leather_boots")
    LEATHER_BOOTS("leather_boots"),
    @XmlEnumValue("chainmail_helmet")
    CHAINMAIL_HELMET("chainmail_helmet"),
    @XmlEnumValue("chainmail_chestplate")
    CHAINMAIL_CHESTPLATE("chainmail_chestplate"),
    @XmlEnumValue("chainmail_leggings")
    CHAINMAIL_LEGGINGS("chainmail_leggings"),
    @XmlEnumValue("chainmail_boots")
    CHAINMAIL_BOOTS("chainmail_boots"),
    @XmlEnumValue("iron_helmet")
    IRON_HELMET("iron_helmet"),
    @XmlEnumValue("iron_chestplate")
    IRON_CHESTPLATE("iron_chestplate"),
    @XmlEnumValue("iron_leggings")
    IRON_LEGGINGS("iron_leggings"),
    @XmlEnumValue("iron_boots")
    IRON_BOOTS("iron_boots"),
    @XmlEnumValue("diamond_helmet")
    DIAMOND_HELMET("diamond_helmet"),
    @XmlEnumValue("diamond_chestplate")
    DIAMOND_CHESTPLATE("diamond_chestplate"),
    @XmlEnumValue("diamond_leggings")
    DIAMOND_LEGGINGS("diamond_leggings"),
    @XmlEnumValue("diamond_boots")
    DIAMOND_BOOTS("diamond_boots"),
    @XmlEnumValue("golden_helmet")
    GOLDEN_HELMET("golden_helmet"),
    @XmlEnumValue("golden_chestplate")
    GOLDEN_CHESTPLATE("golden_chestplate"),
    @XmlEnumValue("golden_leggings")
    GOLDEN_LEGGINGS("golden_leggings"),
    @XmlEnumValue("golden_boots")
    GOLDEN_BOOTS("golden_boots"),
    @XmlEnumValue("flint")
    FLINT("flint"),
    @XmlEnumValue("porkchop")
    PORKCHOP("porkchop"),
    @XmlEnumValue("cooked_porkchop")
    COOKED_PORKCHOP("cooked_porkchop"),
    @XmlEnumValue("painting")
    PAINTING("painting"),
    @XmlEnumValue("golden_apple")
    GOLDEN_APPLE("golden_apple"),
    @XmlEnumValue("sign")
    SIGN("sign"),
    @XmlEnumValue("wooden_door")
    WOODEN_DOOR("wooden_door"),
    @XmlEnumValue("bucket")
    BUCKET("bucket"),
    @XmlEnumValue("water_bucket")
    WATER_BUCKET("water_bucket"),
    @XmlEnumValue("lava_bucket")
    LAVA_BUCKET("lava_bucket"),
    @XmlEnumValue("minecart")
    MINECART("minecart"),
    @XmlEnumValue("saddle")
    SADDLE("saddle"),
    @XmlEnumValue("iron_door")
    IRON_DOOR("iron_door"),
    @XmlEnumValue("redstone")
    REDSTONE("redstone"),
    @XmlEnumValue("snowball")
    SNOWBALL("snowball"),
    @XmlEnumValue("boat")
    BOAT("boat"),
    @XmlEnumValue("leather")
    LEATHER("leather"),
    @XmlEnumValue("milk_bucket")
    MILK_BUCKET("milk_bucket"),
    @XmlEnumValue("brick")
    BRICK("brick"),
    @XmlEnumValue("clay_ball")
    CLAY_BALL("clay_ball"),
    @XmlEnumValue("reeds")
    REEDS("reeds"),
    @XmlEnumValue("paper")
    PAPER("paper"),
    @XmlEnumValue("book")
    BOOK("book"),
    @XmlEnumValue("slime_ball")
    SLIME_BALL("slime_ball"),
    @XmlEnumValue("chest_minecart")
    CHEST_MINECART("chest_minecart"),
    @XmlEnumValue("furnace_minecart")
    FURNACE_MINECART("furnace_minecart"),
    @XmlEnumValue("egg")
    EGG("egg"),
    @XmlEnumValue("compass")
    COMPASS("compass"),
    @XmlEnumValue("fishing_rod")
    FISHING_ROD("fishing_rod"),
    @XmlEnumValue("clock")
    CLOCK("clock"),
    @XmlEnumValue("glowstone_dust")
    GLOWSTONE_DUST("glowstone_dust"),
    @XmlEnumValue("fish")
    FISH("fish"),
    @XmlEnumValue("cooked_fish")
    COOKED_FISH("cooked_fish"),
    @XmlEnumValue("dye")
    DYE("dye"),
    @XmlEnumValue("bone")
    BONE("bone"),
    @XmlEnumValue("sugar")
    SUGAR("sugar"),
    @XmlEnumValue("cake")
    CAKE("cake"),
    @XmlEnumValue("bed")
    BED("bed"),
    @XmlEnumValue("repeater")
    REPEATER("repeater"),
    @XmlEnumValue("cookie")
    COOKIE("cookie"),
    @XmlEnumValue("filled_map")
    FILLED_MAP("filled_map"),
    @XmlEnumValue("shears")
    SHEARS("shears"),
    @XmlEnumValue("melon")
    MELON("melon"),
    @XmlEnumValue("pumpkin_seeds")
    PUMPKIN_SEEDS("pumpkin_seeds"),
    @XmlEnumValue("melon_seeds")
    MELON_SEEDS("melon_seeds"),
    @XmlEnumValue("beef")
    BEEF("beef"),
    @XmlEnumValue("cooked_beef")
    COOKED_BEEF("cooked_beef"),
    @XmlEnumValue("chicken")
    CHICKEN("chicken"),
    @XmlEnumValue("cooked_chicken")
    COOKED_CHICKEN("cooked_chicken"),
    @XmlEnumValue("rotten_flesh")
    ROTTEN_FLESH("rotten_flesh"),
    @XmlEnumValue("ender_pearl")
    ENDER_PEARL("ender_pearl"),
    @XmlEnumValue("blaze_rod")
    BLAZE_ROD("blaze_rod"),
    @XmlEnumValue("ghast_tear")
    GHAST_TEAR("ghast_tear"),
    @XmlEnumValue("gold_nugget")
    GOLD_NUGGET("gold_nugget"),
    @XmlEnumValue("nether_wart")
    NETHER_WART("nether_wart"),
    @XmlEnumValue("potion")
    POTION("potion"),
    @XmlEnumValue("glass_bottle")
    GLASS_BOTTLE("glass_bottle"),
    @XmlEnumValue("spider_eye")
    SPIDER_EYE("spider_eye"),
    @XmlEnumValue("fermented_spider_eye")
    FERMENTED_SPIDER_EYE("fermented_spider_eye"),
    @XmlEnumValue("blaze_powder")
    BLAZE_POWDER("blaze_powder"),
    @XmlEnumValue("magma_cream")
    MAGMA_CREAM("magma_cream"),
    @XmlEnumValue("brewing_stand")
    BREWING_STAND("brewing_stand"),
    @XmlEnumValue("cauldron")
    CAULDRON("cauldron"),
    @XmlEnumValue("ender_eye")
    ENDER_EYE("ender_eye"),
    @XmlEnumValue("speckled_melon")
    SPECKLED_MELON("speckled_melon"),
    @XmlEnumValue("spawn_egg")
    SPAWN_EGG("spawn_egg"),
    @XmlEnumValue("experience_bottle")
    EXPERIENCE_BOTTLE("experience_bottle"),
    @XmlEnumValue("fire_charge")
    FIRE_CHARGE("fire_charge"),
    @XmlEnumValue("writable_book")
    WRITABLE_BOOK("writable_book"),
    @XmlEnumValue("written_book")
    WRITTEN_BOOK("written_book"),
    @XmlEnumValue("emerald")
    EMERALD("emerald"),
    @XmlEnumValue("item_frame")
    ITEM_FRAME("item_frame"),
    @XmlEnumValue("flower_pot")
    FLOWER_POT("flower_pot"),
    @XmlEnumValue("carrot")
    CARROT("carrot"),
    @XmlEnumValue("potato")
    POTATO("potato"),
    @XmlEnumValue("baked_potato")
    BAKED_POTATO("baked_potato"),
    @XmlEnumValue("poisonous_potato")
    POISONOUS_POTATO("poisonous_potato"),
    @XmlEnumValue("map")
    MAP("map"),
    @XmlEnumValue("golden_carrot")
    GOLDEN_CARROT("golden_carrot"),
    @XmlEnumValue("skull")
    SKULL("skull"),
    @XmlEnumValue("carrot_on_a_stick")
    CARROT_ON_A_STICK("carrot_on_a_stick"),
    @XmlEnumValue("nether_star")
    NETHER_STAR("nether_star"),
    @XmlEnumValue("pumpkin_pie")
    PUMPKIN_PIE("pumpkin_pie"),
    @XmlEnumValue("fireworks")
    FIREWORKS("fireworks"),
    @XmlEnumValue("firework_charge")
    FIREWORK_CHARGE("firework_charge"),
    @XmlEnumValue("enchanted_book")
    ENCHANTED_BOOK("enchanted_book"),
    @XmlEnumValue("comparator")
    COMPARATOR("comparator"),
    @XmlEnumValue("netherbrick")
    NETHERBRICK("netherbrick"),
    @XmlEnumValue("quartz")
    QUARTZ("quartz"),
    @XmlEnumValue("tnt_minecart")
    TNT_MINECART("tnt_minecart"),
    @XmlEnumValue("hopper_minecart")
    HOPPER_MINECART("hopper_minecart"),
    @XmlEnumValue("prismarine_shard")
    PRISMARINE_SHARD("prismarine_shard"),
    @XmlEnumValue("prismarine_crystals")
    PRISMARINE_CRYSTALS("prismarine_crystals"),
    @XmlEnumValue("rabbit")
    RABBIT("rabbit"),
    @XmlEnumValue("cooked_rabbit")
    COOKED_RABBIT("cooked_rabbit"),
    @XmlEnumValue("rabbit_stew")
    RABBIT_STEW("rabbit_stew"),
    @XmlEnumValue("rabbit_foot")
    RABBIT_FOOT("rabbit_foot"),
    @XmlEnumValue("rabbit_hide")
    RABBIT_HIDE("rabbit_hide"),
    @XmlEnumValue("armor_stand")
    ARMOR_STAND("armor_stand"),
    @XmlEnumValue("iron_horse_armor")
    IRON_HORSE_ARMOR("iron_horse_armor"),
    @XmlEnumValue("golden_horse_armor")
    GOLDEN_HORSE_ARMOR("golden_horse_armor"),
    @XmlEnumValue("diamond_horse_armor")
    DIAMOND_HORSE_ARMOR("diamond_horse_armor"),
    @XmlEnumValue("lead")
    LEAD("lead"),
    @XmlEnumValue("name_tag")
    NAME_TAG("name_tag"),
    @XmlEnumValue("command_block_minecart")
    COMMAND_BLOCK_MINECART("command_block_minecart"),
    @XmlEnumValue("mutton")
    MUTTON("mutton"),
    @XmlEnumValue("cooked_mutton")
    COOKED_MUTTON("cooked_mutton"),
    @XmlEnumValue("banner")
    BANNER("banner"),
    @XmlEnumValue("spruce_door")
    SPRUCE_DOOR("spruce_door"),
    @XmlEnumValue("birch_door")
    BIRCH_DOOR("birch_door"),
    @XmlEnumValue("jungle_door")
    JUNGLE_DOOR("jungle_door"),
    @XmlEnumValue("acacia_door")
    ACACIA_DOOR("acacia_door"),
    @XmlEnumValue("dark_oak_door")
    DARK_OAK_DOOR("dark_oak_door"),
    @XmlEnumValue("chorus_fruit")
    CHORUS_FRUIT("chorus_fruit"),
    @XmlEnumValue("chorus_fruit_popped")
    CHORUS_FRUIT_POPPED("chorus_fruit_popped"),
    @XmlEnumValue("beetroot")
    BEETROOT("beetroot"),
    @XmlEnumValue("beetroot_seeds")
    BEETROOT_SEEDS("beetroot_seeds"),
    @XmlEnumValue("beetroot_soup")
    BEETROOT_SOUP("beetroot_soup"),
    @XmlEnumValue("dragon_breath")
    DRAGON_BREATH("dragon_breath"),
    @XmlEnumValue("splash_potion")
    SPLASH_POTION("splash_potion"),
    @XmlEnumValue("spectral_arrow")
    SPECTRAL_ARROW("spectral_arrow"),
    @XmlEnumValue("tipped_arrow")
    TIPPED_ARROW("tipped_arrow"),
    @XmlEnumValue("lingering_potion")
    LINGERING_POTION("lingering_potion"),
    @XmlEnumValue("shield")
    SHIELD("shield"),
    @XmlEnumValue("elytra")
    ELYTRA("elytra"),
    @XmlEnumValue("spruce_boat")
    SPRUCE_BOAT("spruce_boat"),
    @XmlEnumValue("birch_boat")
    BIRCH_BOAT("birch_boat"),
    @XmlEnumValue("jungle_boat")
    JUNGLE_BOAT("jungle_boat"),
    @XmlEnumValue("acacia_boat")
    ACACIA_BOAT("acacia_boat"),
    @XmlEnumValue("dark_oak_boat")
    DARK_OAK_BOAT("dark_oak_boat"),
    @XmlEnumValue("totem_of_undying")
    TOTEM_OF_UNDYING("totem_of_undying"),
    @XmlEnumValue("shulker_shell")
    SHULKER_SHELL("shulker_shell"),
    @XmlEnumValue("iron_nugget")
    IRON_NUGGET("iron_nugget"),
    @XmlEnumValue("record_13")
    RECORD_13("record_13"),
    @XmlEnumValue("record_cat")
    RECORD_CAT("record_cat"),
    @XmlEnumValue("record_blocks")
    RECORD_BLOCKS("record_blocks"),
    @XmlEnumValue("record_chirp")
    RECORD_CHIRP("record_chirp"),
    @XmlEnumValue("record_far")
    RECORD_FAR("record_far"),
    @XmlEnumValue("record_mall")
    RECORD_MALL("record_mall"),
    @XmlEnumValue("record_mellohi")
    RECORD_MELLOHI("record_mellohi"),
    @XmlEnumValue("record_stal")
    RECORD_STAL("record_stal"),
    @XmlEnumValue("record_strad")
    RECORD_STRAD("record_strad"),
    @XmlEnumValue("record_ward")
    RECORD_WARD("record_ward"),
    @XmlEnumValue("record_11")
    RECORD_11("record_11"),
    @XmlEnumValue("record_wait")
    RECORD_WAIT("record_wait");
    private final String value;

    ItemType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static ItemType fromValue(String v) {
        for (ItemType c: ItemType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
