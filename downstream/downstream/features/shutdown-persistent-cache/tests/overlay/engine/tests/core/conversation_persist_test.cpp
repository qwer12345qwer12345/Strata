// The shutdown persistent cache (conversation_persist.hpp): a normal end's flush, the next start's scan, the
// restore's non-destructive read, the namespace isolation and the fail-safe behaviour.  No GPU, no model.
#include "strata/core/conversation_persist.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
// like the spill test's: a parked conversation whose running state is `fill`-filled bytes, an optional checkpoint
auto img = [](std::vector<int32_t> ids, uint8_t fill, size_t state = 1 << 20, std::vector<int32_t> ckpt = {}) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.assign(state, fill);
    s.cvec = true;
    if (!ckpt.empty()) {
        ConversationCheckpoint c;
        c.ids = ckpt;
        c.gdn.assign(4096, (uint8_t) (fill + 1));
        s.checkpoints.push_back(c);
    }
    return s;
};
// every scenario gets its own configuration hash, so its namespace folder is its own (and so the isolation itself
// is exercised by the test's structure)
const SessionFileIdentity id_main{11, 22}, id_other{11, 23}, id_chain{11, 24}, id_budget{11, 25}, id_refuse{11, 26},
    id_lock{11, 27};
}  // namespace

int main() {
    std::error_code ec;
    const fs::path root =
        fs::temp_directory_path() / ("strata-persist-test-" + std::to_string(std::random_device{}()));
    fs::remove_all(root, ec);
    const uint64_t cap = (uint64_t(9) << 20);   // room for several ~1 MiB payloads with their margins

    // the flush: every entry of a normal end's RAM cache goes, whole, into the identity's own folder
    ShutdownPersist a;
    std::string err;
    SavedConversation one = img({1, 2, 3, 4}, 7, 1 << 20, {1, 2}), two = img({5, 6, 7}, 9),
                      three = img({20, 21, 22, 23}, 5, 1 << 20, {20, 21});
    const std::vector<const SavedConversation*> images{&one, &two, &three};
    check(a.open(root, id_main, cap, err) && a.enabled(), "the tier opens");
    check(fs::is_directory(root / "conversation-persist"), "the persistent root is made");
    check(a.save_all(images) == 3 && a.size() == 3 && a.bytes() > 0, "three conversations persisted");
    check(a.bytes() == a.entry(0)->bytes + a.entry(1)->bytes + a.entry(2)->bytes, "the budget counts what is written");
    for (size_t i = 0; i < 3; ++i)
        check(fs::is_regular_file(a.entry(i)->file) && fs::file_size(a.entry(i)->file) == a.entry(i)->bytes &&
              fs::is_regular_file(fs::path(a.entry(i)->file).replace_extension(".meta")),
              "each record has its payload and its sidecar");
    check(a.save_all({}) == 0, "nothing to write: nothing written");

    // the next start: the same namespace scans back in from the sidecars alone (no payload read yet)
    {
        ShutdownPersist b;
        check(b.open(root, id_main, cap, err) && b.size() == 3, "a restart indexes the records of the last one");
        check(b.entry(0)->serial == a.entry(0)->serial && b.entry(0)->bytes == a.entry(0)->bytes &&
              b.entry(0)->cvec == a.entry(0)->cvec, "the scan sees the same records");
        check(b.entry(0)->live.ids == std::vector<int32_t>({1, 2, 3, 4}) &&
              b.entry(0)->checkpoints.size() == 1 && b.entry(0)->checkpoints[0].ids == std::vector<int32_t>({1, 2}),
              "the stubs carry the ids of the live state and of the checkpoint");
        // the restore: a prompt that continues a record reads it whole, byte for byte, and consumes nothing
        const auto m = b.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true);
        check(m.tokens == 4 && m.live, "a prompt finds the record it continues");
        const auto mc = b.best(std::vector<int32_t>{20, 21, 99}, {}, true);
        check(mc.tokens == 2 && !mc.live, "a prompt that shares only a checkpoint finds it");
        check(b.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, false).tokens == 0, "other steering: none");
        check(b.best(std::vector<int32_t>{31, 32}, {}, true).tokens == 0, "a prompt nothing starts: none");
        SavedConversation back;
        check(b.read_serial(m.serial, {}, back, err) && back.live.ids == std::vector<int32_t>({1, 2, 3, 4}) &&
              back.live.gdn == std::vector<uint8_t>(1 << 20, 7) &&
              back.checkpoints.size() == 1 && back.checkpoints[0].gdn == std::vector<uint8_t>(4096, 8),
              "read back byte for byte");
        check(b.size() == 3 && b.find(m.serial), "a restore does not consume its record");
        SavedConversation again;
        check(b.read_serial(m.serial, {}, again, err) && again.live.gdn == std::vector<uint8_t>(1 << 20, 7),
              "the record answers the next restart too");
        // a record that no longer reads back is dropped, alone, and the miss changes nothing else
        const fs::path payload = b.entry(2)->file;
        fs::resize_file(payload, fs::file_size(payload) / 2, ec);
        SavedConversation x;
        check(!ec && !b.read_serial(b.entry(2)->serial, {}, x, err) && b.size() == 2 &&
              !fs::exists(payload) && !fs::exists(fs::path(payload).replace_extension(".meta")),
              "a payload that cannot be read is dropped with its sidecar");
        check(b.read_serial(m.serial, {}, x, err), "the other records are untouched");
    }

    // another model's (or configuration's) namespace: its own folder, no shared records
    {
        SavedConversation four = img({1, 2, 3, 4}, 3);
        ShutdownPersist other;
        check(other.open(root, id_other, cap, err) && other.size() == 0, "another config: an empty index");
        check(other.save_all({&four}) == 1, "it writes its own");
        check(other.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true).tokens == 4, "its own records answer it");
        ShutdownPersist a2;
        check(a2.open(root, id_main, cap, err) && a2.size() == 2, "the first identity does not see the other's record");
        SavedConversation back_a, back_o;
        const auto ma = a2.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true);
        const auto mo = other.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true);
        check(a2.read_serial(ma.serial, {}, back_a, err) && back_a.live.gdn == std::vector<uint8_t>(1 << 20, 7),
              "the first identity reads its own conversation");
        check(other.read_serial(mo.serial, {}, back_o, err) && back_o.live.gdn == std::vector<uint8_t>(1 << 20, 3),
              "the other identity reads its own, never the first's");
    }

    // a new snapshot of the same conversation retires the old record - only once the new one is complete - and a
    // sibling that only shares a prefix keeps its record
    {
        ShutdownPersist c;
        check(c.open(root, id_chain, cap, err) && c.size() == 0, "the chain scenario starts from its own namespace");
        SavedConversation parent = img({1, 2, 3, 4}, 7, 1 << 20, {1, 2});
        check(c.save_all({&parent}) == 1 && c.size() == 1, "the parent persists");
        const fs::path parent_payload = c.entry(0)->file;
        SavedConversation newer = img({1, 2, 3, 4}, 1, 1 << 20, {1, 2});
        check(c.save_all({&newer}) == 1 && c.size() == 1, "a new snapshot of the same conversation replaces the old one");
        check(!fs::exists(parent_payload), "the old record's files went once the new one was complete");
        SavedConversation back;
        const auto m = c.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true);
        check(c.read_serial(m.serial, {}, back, err) && back.live.gdn == std::vector<uint8_t>(1 << 20, 1),
              "the newest snapshot is the one a restart reads");
        SavedConversation sibling = img({1, 2, 3, 4, 30, 31}, 6);
        check(c.save_all({&sibling}) == 1 && c.size() == 2, "a sibling conversation persists beside it");
        SavedConversation third = img({1, 2, 3, 4, 40}, 8);
        check(c.save_all({&third}) == 1 && c.size() == 3, "a third conversation persists beside both");
        const auto ms = c.best(std::vector<int32_t>{1, 2, 3, 4, 30, 31, 1}, {}, true);
        SavedConversation back_s;
        check(ms.tokens == 6 && c.read_serial(ms.serial, {}, back_s, err) &&
              back_s.live.gdn == std::vector<uint8_t>(1 << 20, 6),
              "the sibling's record survives another conversation's flush");
    }

    // the budget: an image larger than the whole budget is refused alone; a third record pushes the oldest out,
    // whole; a lowered budget retires at the scan.  (The write's own margin is one MiB, like the spill tier's, so
    // the records here are sized so one more record fits exactly when one whole record goes.)
    {
        ShutdownPersist big;
        check(big.open(root, id_budget, (uint64_t(1) << 20), err), "a one-MiB budget opens");
        SavedConversation huge = img({1, 2, 3, 4, 5, 6, 7, 8}, 4);
        check(big.save_all({&huge}) == 0 && big.size() == 0, "an image larger than the whole budget is refused");
        ShutdownPersist grow;
        check(grow.open(root, id_budget, cap, err), "the budget scenario reopens with room");
        SavedConversation fit = img({60, 61}, 2, 3 << 20), push = img({62, 63}, 3, 3 << 20);
        check(grow.save_all({&fit, &push}) == 2 && grow.size() == 2, "two records fit");
        const uint64_t two = grow.bytes();
        grow.close();
        ShutdownPersist small;
        check(small.open(root, id_budget, two, err) && small.size() == 2, "a budget of exactly the two keeps both");
        const uint64_t one = small.entry(0)->bytes;
        SavedConversation third = img({64, 65}, 4, 1536 << 10);
        check(small.save_all({&third}) == 1 && small.size() == 2 &&
              small.best(std::vector<int32_t>{64, 65, 1}, {}, true).tokens == 2 &&
              small.best(std::vector<int32_t>{60, 61, 1}, {}, true).tokens == 0,
              "a third record pushes the oldest out, whole");
        small.close();
        ShutdownPersist lowered;
        check(lowered.open(root, id_budget, one + (64 << 10), err) && lowered.size() == 1 &&
              lowered.best(std::vector<int32_t>{64, 65, 1}, {}, true).tokens == 2,
              "a lowered budget retires at the scan and keeps the newest");
    }

    // a damaged namespace costs its records, never the tier: a cut sidecar, a foreign file and a missing payload
    // are ignored, alone, and the scan deletes nothing it does not own
    {
        ShutdownPersist d;
        check(d.open(root, id_main, cap, err) && d.size() == 2, "the main namespace as the runs left it");
        const fs::path sidecar = fs::path(d.entry(0)->file).replace_extension(".meta");
        fs::resize_file(sidecar, fs::file_size(sidecar) / 3, ec);
        check(!ec, "the sidecar is cut");
        std::ofstream(root / "conversation-persist" / "persist-999-aaaaaaaa.meta", std::ios::binary) << "not a sidecar";
        const fs::path doomed = d.entry(1)->file;
        fs::remove(doomed, ec);
        ShutdownPersist e;
        check(e.open(root, id_main, cap, err) && e.size() == d.size() - 2,
              "a cut sidecar and a missing payload are ignored");
        bool foreign = false;
        for (auto& x : fs::directory_iterator(root / "conversation-persist"))
            if (x.path().filename() == "persist-999-aaaaaaaa.meta") foreign = true;
        check(foreign, "nothing of the scan's business is deleted by a scan");
        // (the sidecar cut above belongs to the record a block-4 read dropped; the namespace holds one record)
    }

    // a refused image leaves nothing behind; an empty live state and a layer split are not conversations
    {
        ShutdownPersist f;
        check(f.open(root, id_refuse, cap, err) && f.size() == 0, "the refuse scenario's own namespace opens");
        SavedConversation empty;
        empty.live.gdn.assign(1 << 20, 5);
        SavedConversation split = img({1, 2}, 5);
        split.stage_images.emplace_back();
        check(f.save_all({&empty, &split}) == 0 && f.size() == 0, "an empty live state and a layer split are refused");
        size_t left = 0;
        for (auto& x : fs::directory_iterator(f.dir()))
            if (x.path().filename() != ".lock") ++left;
        check(left == 0, "nothing of a refused flush is left on the disk");
    }

    // two engines shutting down together: the one that holds the namespace lock flushes, the other's flush is a
    // clean no-op
    {
        ShutdownPersist g;
        SavedConversation mine = img({70, 71}, 2), theirs = img({72, 73}, 3);
        check(g.open(root, id_lock, cap, err) && !g.dir().empty(), "the first engine opens its namespace");
        check(g.save_all({&mine}) == 1, "the first engine writes before anyone holds the lock");
        ShutdownPersist h;
        check(h.open(root, id_lock, cap, err) && h.size() == 1, "the second engine opens (a scan needs no lock)");
        const spill_detail::LockHandle held = spill_detail::try_lock(g.dir() / ".lock");
        check(held != spill_detail::no_lock(), "a second engine's namespace lock is taken by the test");
        std::vector<const SavedConversation*> both{&mine, &theirs};
        check(h.save_all(both) == 0 && h.size() == 1, "the other engine's flush writes nothing while the lock is held");
        spill_detail::unlock(held);
        check(g.save_all(both) == 2, "once free, the flush goes");
    }

    fs::remove_all(root, ec);
    std::printf("conversation_persist_test: %d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
